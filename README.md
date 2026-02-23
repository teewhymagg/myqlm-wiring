# Network Routing QUBO PoC

Proof of concept for solving a 6-node network routing problem using the QUBO (Quadratic Unconstrained Binary Optimization) model.

The goal is to find the lowest-cost path between a source node (A) and a destination node (F) on a defined graph without using slack variables. We use Flow Conservation constraints (the "House of Santa Claus" principle) on directed edges to ensure a continuous path.

## Setup

Requires python 3.10+.
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Files
* `graph.json`: Defines the 6 nodes and 8 edges with their respective cable lengths (costs).
* `qubo_builder.py`: Python class that translates the graph and flow conservation math into a 16x16 symmetric Q matrix.
* `brute_force.py`: Scans all 65,536 combinations to find the exact global minimum energy state. Proves the math formulation.
* `solve.py`: Uses a Custom Simulated Annealing heuristic to find a solution on the Q matrix.

## Running

To run the exact solver:
```bash
python brute_force.py
```

To run the quantum heuristic simulator:
```bash
python solve.py
```
