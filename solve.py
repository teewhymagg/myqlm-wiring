import json
import numpy as np
from qubo_builder import RoutingQUBO
from qat.core import Result
from qat.opt import QUBO
from qat.qpus import SimulatedAnnealing
import time

def load_graph(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['nodes'], data['edges']

class SimulatedAnnealingSolver:
    def __init__(self, temp_max=100.0, temp_min=0.001, n_steps=50000, seed=42, n_restarts=20):
        self.temp_max = temp_max
        self.temp_min = temp_min
        self.n_steps = n_steps
        self.seed = seed
        self.n_restarts = n_restarts

    def solve(self, q_matrix, offset):
        def evaluate(state):
            return float(state @ q_matrix @ state + offset)

        n_vars = len(q_matrix)
        best_solution = None
        best_energy = float("inf")

        for restart in range(self.n_restarts):
            np.random.seed(self.seed + restart * 137)
            state = np.random.randint(0, 2, size=n_vars)
            energy = evaluate(state)

            if energy < best_energy:
                best_energy = energy
                best_solution = state.copy()

            t_factor = (self.temp_min / self.temp_max) ** (1.0 / self.n_steps)
            temp = self.temp_max

            for step in range(self.n_steps):
                idx = np.random.randint(n_vars)
                old_val = state[idx]
                delta_x = 1 if old_val == 0 else -1
                
                delta_e = 2 * delta_x * np.dot(q_matrix[idx, :], state) + q_matrix[idx, idx] * (delta_x ** 2)
                if delta_e < 0 or np.random.random() < np.exp(-delta_e / temp):
                    state[idx] = 1 - old_val
                    energy += delta_e
                    if energy < best_energy:
                        best_energy = energy
                        best_solution = state.copy()
                temp *= t_factor

        return {"solution": best_solution, "energy": best_energy}

def solve_with_sqa(Q, offset, builder, num_sweeps=20000):
    print(f"Running SQA on {builder.num_vars} qubits")
    
    solver = SimulatedAnnealingSolver(n_steps=num_sweeps)
    start_time = time.time()
    result = solver.solve(Q, offset)
    print(f"SQA completed in {time.time() - start_time:.2f} seconds.")
    
    best_state = result["solution"]
    best_energy = result["energy"]
    
    binary_solution = [int(v) for v in best_state]
    path, cost = builder.decode(binary_solution)
    
    valid = (best_energy < builder.P)
    
    print("\nSQA Best Solution")
    print(f"Energy: {best_energy:.4f}")
    print(f"Status: {'VALID' if valid else 'INVALID'}")
    print(f"Path: {path}")
    print(f"Cost: {cost:.4f}")

if __name__ == "__main__":
    nodes, edges = load_graph('graph.json')
    source = 'A'
    dest = 'F'
    penalty = 100.0  
    
    builder = RoutingQUBO(nodes, edges, source, dest, penalty_weight=penalty)
    Q, offset = builder.build_qubo()
    
    solve_with_sqa(Q, offset, builder)
