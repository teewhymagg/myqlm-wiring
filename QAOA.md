# Solving the Routing Problem with QAOA

## What is QAOA?

QAOA (Quantum Approximate Optimization Algorithm) is a variational quantum algorithm.
It takes an optimization problem, encodes it as a quantum Hamiltonian, and searches for
the lowest-energy state by running a parameterized quantum circuit and tuning its
parameters classically.

The algorithm alternates between two operations for `p` layers (the circuit depth):

1. **Cost unitary** — applies phases proportional to the QUBO energy, reinforcing states
   with low energy
2. **Mixer unitary** — applies X rotations to each qubit, allowing the quantum state to
   "explore" neighbouring bit strings through quantum superposition

Each layer has one angle `γ` (cost) and one angle `β` (mixer). A depth-`p` circuit
has `2p` free parameters in total.

---

## How the QUBO maps onto the circuit

The QUBO matrix M and offset from `qubo_builder.py` describe the objective:

```
energy = x^T M x + offset
```

myQLM's `QUBO` class minimises `H = -x^T Q x - E_Q`, so we pass `Q = -M` and
`offset_q = -offset` — the same sign flip used in `solve.py`.

The cost Hamiltonian is built from `get_observable()`, which converts each diagonal
and off-diagonal term of M into Z and ZZ Pauli operators:

- A diagonal term `M[i,i]` becomes a `Z_i` rotation
- An off-diagonal term `M[i,j]` becomes a `Z_i Z_j` rotation

With 16 variables, M is 56% dense, giving ~100 distinct ZZ interaction terms.
Using `cnots=False` the circuit encodes each ZZ term as a single multi-qubit rotation
gate (208 gates at depth=2), rather than decomposing into CNOT pairs (464 gates).
This halves the circuit size with no loss in expressiveness.

---

## Step 1 — Build the QAOA ansatz

```python
job = qubo_problem.qaoa_ansatz(depth=2, cnots=False)
```

This returns a parameterized Job. Its circuit contains:

```
H^⊗16                             (initial superposition: all 2^16 states equally likely)
for each of depth=2 layers:
    cost unitary  (parameterised by γ_k)
    mixer unitary (parameterised by β_k)
```

The 4 free parameters are `γ_0, γ_1, β_0, β_1`.

---

## Step 2 — Optimize the parameters

```python
plugin = ScipyMinimizePlugin(method="COBYLA", tol=1e-2, options={"maxiter": 20})
opt_result = (plugin | PyLinalg()).submit(job)
```

`ScipyMinimizePlugin` repeatedly evaluates the expectation value `<ψ(β,γ)|H|ψ(β,γ)>`
and updates the angles to minimize it. Each evaluation simulates the full 16-qubit
statevector using `PyLinalg` (a classical statevector simulator).

With 16 qubits (2^16 = 65536 amplitudes) and 208 gates, each evaluation takes ~5s on
a laptop. The optimizer is capped at 20 iterations (~100s total) rather than waiting
for full convergence, because:

- Full COBYLA convergence would take ~500 iterations (~2500s)
- 20 steps already move the parameter landscape meaningfully away from random
- The greedy post-processing (Step 4) does the heavy lifting for solution quality

---

## Step 3 — Sample the optimized circuit

```python
bound_circuit = job.circuit(**optimal_params)
sample_result = PyLinalg().submit(Job(circuit=bound_circuit, nbshots=200))
```

With the parameters fixed, the circuit produces a probability distribution over all
2^16 = 65536 bit strings. We draw 200 samples from it. Each sample is a 16-bit string
representing a candidate assignment of the edge variables.

**Bit ordering:** myQLM uses MSB convention — qubit 0 maps to the most significant bit.
Decoding: `x_i = (state_int >> (n-1-i)) & 1`

Most samples will have high QUBO energy (invalid assignments). The value of the QAOA
circuit is to produce a distribution *biased toward lower-energy regions* compared to
uniform random sampling, giving the greedy post-processing better starting points.

---

## Step 4 — Post-process each sample

Each of the 200 samples goes through the same pipeline as in `solve.py`:

```
sample bitstring
      ↓
multi-start randomised greedy descent   (find nearest local minimum)
      ↓
cycle removal                           (strip back-and-forth edge pairs)
      ↓
multi-start randomised greedy descent   (tidy up after removal)
      ↓
check: is this a valid simple path?
```

The best valid result across all 200 samples is kept.

---

## Why not just optimize to convergence?

The problem has a large penalty term (P=15) that dwarfs the path cost differences
(4.3 vs 4.6). This makes the QUBO energy landscape extremely rough — most states have
energy in the range 30–200, while valid paths sit at 4–9. QAOA at low depth cannot
concentrate amplitude on valid states alone; the expectation value (111 at convergence)
is dominated by the many invalid states.

The greedy post-processing compensates for this. Because:

1. QAOA samples are spread across a wide region of the energy landscape
2. Even states with energy 30–150 can be within a few greedy steps of a valid path
3. The multi-start randomised descent finds both local minima (A→B→E→F at 4.6) and
   the global minimum (A→D→F at 4.3) depending on the random flip ordering

QAOA's role is therefore to generate *diverse starting points* for greedy descent,
not to directly solve the problem.

---

## Comparison with SA (`solve.py`)

| | Simulated Annealing | QAOA |
|---|---|---|
| Exploration mechanism | Random spin flips at temperature T | Quantum superposition + variational angles |
| Samples generated | 1 per SA run (30 runs) | 200 from one optimized circuit |
| Time | ~5s | ~100s |
| Simulation engine | `SimulatedAnnealing` QPU | `PyLinalg` statevector simulator |
| On real quantum hardware | Not applicable | Would take milliseconds per shot |
| Post-processing | Multi-greedy + cycle removal | Same |
| Result | A → D → F, 4.3 m | A → D → F, 4.3 m |

The SA approach is much faster in classical simulation. On actual quantum hardware, the
QAOA circuit evaluation would be near-instantaneous, making QAOA the more scalable
approach as problem size grows.

---

## Result

```
optimizing QAOA parameters (20 steps, depth=2)...
done in 94.8s  expectation=89.286
sampling 200 shots...
done in 0.6s
total: 99.4s
path:   A -> D -> F
cost:   4.3000 m
energy: 4.3000
```

The QAOA solver finds the globally optimal path **A → D → F** at **4.3 m**, matching
the brute-force result.
