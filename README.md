# Network Routing QUBO

Proof of concept for solving a 6-node network routing problem using QUBO (Quadratic Unconstrained Binary Optimization), solved with myQLM's Simulated Annealing and QAOA.

The goal is to find the lowest-cost path between source node **A** and destination node **F** on a directed graph. Flow conservation constraints enforce a continuous, loop-free path without slack variables.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Files

| File | Description |
|---|---|
| `graph.json` | 6 nodes, 8 edges with cable lengths (costs) |
| `qubo_builder.py` | Builds the 16×16 QUBO matrix from the graph and flow conservation constraints |
| `brute_force.py` | Exhaustive search over all 65 536 combinations — proves the formulation is correct |
| `solve.py` | Solves using myQLM **Simulated Annealing** (SA) |
| `qaoa_solve.py` | Solves using myQLM **QAOA** (variational quantum circuit) |
| `SOLUTION_EXPLAINED.md` | Plain-language walkthrough of the SA solver and all post-processing steps |
| `QAOA.md` | Plain-language walkthrough of the QAOA solver |

## Running

```bash
# exact brute-force baseline
python brute_force.py

# SA solver (myQLM SimulatedAnnealing + greedy post-processing, ~5s)
python solve.py

# QAOA solver (myQLM PyLinalg statevector simulation, ~100s)
python qaoa_solve.py
```

## Approach

Both solvers share the same QUBO formulation from `qubo_builder.py`. Each directed edge becomes a binary variable; the matrix encodes edge costs on the diagonal and flow-conservation penalties on the off-diagonal terms. Any valid simple path from A to F has energy equal to its total cable length; invalid states carry an additional penalty P = 15.

**SA solver (`solve.py`)** runs 30 restarts of myQLM's `SimulatedAnnealing` QPU. Each restart is followed by a multi-start randomised greedy descent and a back-and-forth cycle removal pass to push the SA result into the nearest valid local minimum.

**QAOA solver (`qaoa_solve.py`)** builds a depth-2 QAOA ansatz via `qubo_problem.qaoa_ansatz()`, partially optimizes the variational angles with COBYLA (20 steps), then draws 200 samples from the optimized circuit. The same greedy post-processing pipeline is applied to each sample.

Both solvers find the globally optimal route: **A → D → F** at **4.3 m**.
