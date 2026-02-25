import json
import warnings
import time

import numpy as np

from qubo_builder import RoutingQUBO
from qat.opt import QUBO
from qat.plugins import ScipyMinimizePlugin
from qat.qpus import PyLinalg
from qat.core import Job


def load_graph(filepath: str):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["nodes"], data["edges"]


def qubo_energy(x, M: np.ndarray, offset: float) -> float:
    x = np.asarray(x, dtype=float)
    return float(x @ M @ x) + offset


def greedy_descent(x: np.ndarray, M: np.ndarray, offset: float, rng=None):
    x = np.array(x, dtype=int)
    e = qubo_energy(x, M, offset)
    while True:
        improvements = []
        for i in range(len(x)):
            x[i] ^= 1
            delta = qubo_energy(x, M, offset) - e
            if delta < -1e-9:
                improvements.append((i, delta))
            x[i] ^= 1
        if not improvements:
            break
        if rng is None:
            best_i, best_delta = min(improvements, key=lambda t: t[1])
        else:
            best_i, best_delta = improvements[rng.integers(len(improvements))]
        x[best_i] ^= 1
        e += best_delta
    return x, e


def multi_greedy_descent(x: np.ndarray, M: np.ndarray, offset: float, n_attempts: int = 20, seed: int = 0):
    best_x, best_e = greedy_descent(x, M, offset)
    rng = np.random.default_rng(seed)
    for _ in range(n_attempts - 1):
        xr, er = greedy_descent(x, M, offset, rng=rng)
        if er < best_e - 1e-9:
            best_e = er
            best_x = xr
    return best_x, best_e


def remove_back_forth_pairs(x: np.ndarray, builder: RoutingQUBO) -> np.ndarray:
    x = x.copy()
    changed = True
    while changed:
        changed = False
        for i, (u, v) in enumerate(builder.vars):
            if x[i] == 1:
                j = builder.var_to_idx.get((v, u), -1)
                if j >= 0 and x[j] == 1:
                    x[i] = 0
                    x[j] = 0
                    changed = True
    return x


def is_valid_simple_path(path: list[tuple], source: str, dest: str) -> bool:
    if not path:
        return False
    if path[0][0] != source or path[-1][1] != dest:
        return False
    visited = set()
    for i, (u, v) in enumerate(path):
        if i < len(path) - 1 and path[i][1] != path[i + 1][0]:
            return False
        if u in visited:
            return False
        visited.add(u)
    return True


def decode_state(state_int: int, n_vars: int) -> np.ndarray:
    # QAOA uses MSB ordering: qubit 0 is the most significant bit
    return np.array([(state_int >> (n_vars - 1 - i)) & 1 for i in range(n_vars)], dtype=int)


def solve_with_myqlm_qaoa(
    nodes: list[str],
    edges: list[dict],
    source: str,
    dest: str,
    *,
    penalty: float = 15.0,
    depth: int = 2,
    n_optimizer_steps: int = 20,
    n_samples: int = 200,
    n_greedy_attempts: int = 20,
    seed: int = 0,
):
    builder = RoutingQUBO(nodes, edges, source, dest, penalty_weight=penalty)
    M, offset = builder.build_qubo()
    n = builder.num_vars

    # myQLM minimises H = -x^T Q x - E_Q
    qubo_problem = QUBO(Q=-M, offset_q=-offset)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        job = qubo_problem.qaoa_ansatz(depth=depth, cnots=False)

    # partially optimize variational parameters (full convergence is slow for 16 qubits)
    plugin = ScipyMinimizePlugin(
        method="COBYLA",
        tol=1e-2,
        options={"maxiter": n_optimizer_steps},
    )

    print(f"optimizing QAOA parameters ({n_optimizer_steps} steps, depth={depth})...")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt_result = (plugin | PyLinalg()).submit(job)

    print(f"done in {time.time() - t0:.1f}s  expectation={opt_result.value:.3f}")

    # bind optimal parameters and sample the circuit
    optimal_params = {k: v.real for k, v in opt_result.parameter_map.items()}
    bound_circuit = job.circuit(**optimal_params)
    sampling_job = Job(circuit=bound_circuit, nbshots=n_samples)

    print(f"sampling {n_samples} shots...")
    t1 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample_result = PyLinalg().submit(sampling_job)
    print(f"done in {time.time() - t1:.1f}s")

    # post-process each sample with multi-greedy descent + cycle removal
    best_energy = float("inf")
    best_path = None
    best_cost = float("inf")

    rng = np.random.default_rng(seed)
    for idx, s in enumerate(sample_result.raw_data):
        x = decode_state(s.state.int, n)

        xg, _ = multi_greedy_descent(x, M, offset, n_attempts=n_greedy_attempts, seed=int(rng.integers(1 << 31)))
        xc = remove_back_forth_pairs(xg, builder)
        xc, e_final = multi_greedy_descent(xc, M, offset, n_attempts=n_greedy_attempts, seed=int(rng.integers(1 << 31)))

        path, cost = builder.decode(xc.tolist())
        valid = is_valid_simple_path(path, source, dest)

        if valid and e_final < best_energy:
            best_energy = e_final
            best_path = path
            best_cost = cost

    elapsed = time.time() - t0
    print(f"total: {elapsed:.1f}s")

    if best_path:
        route = " -> ".join([e[0] for e in best_path] + [best_path[-1][1]])
        print(f"path:   {route}")
        print(f"cost:   {best_cost:.4f} m")
        print(f"energy: {best_energy:.4f}")
    else:
        print("no valid path found, try increasing n_samples or n_greedy_attempts")

    return best_path, best_cost, best_path is not None


if __name__ == "__main__":
    nodes, edges = load_graph("graph.json")

    solve_with_myqlm_qaoa(
        nodes, edges,
        source="A",
        dest="F",
        penalty=15.0,
        depth=2,
        n_optimizer_steps=20,
        n_samples=200,
        n_greedy_attempts=20,
    )
