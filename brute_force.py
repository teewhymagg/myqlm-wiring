import json
import itertools
import numpy as np
import time
from qubo_builder import RoutingQUBO

def load_graph(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['nodes'], data['edges']

def evaluate_qubo(Q, offset, x):
    x_vec = np.array(x)
    return np.dot(x_vec, np.dot(Q, x_vec)) + offset

def solve_brute_force(Q, offset, num_vars):
    print(f"Brute-forcing {num_vars} variables ({2**num_vars} combinations)...")
    start_time = time.time()
    
    best_energy = float('inf')
    best_solutions = []
    
    combinations = itertools.product([0, 1], repeat=num_vars)
    
    for x in combinations:
        energy = evaluate_qubo(Q, offset, x)
        if energy < best_energy:
            best_energy = energy
            best_solutions = [x]
        elif energy == best_energy:
            best_solutions.append(x)
            
    print(f"Done in {time.time() - start_time:.2f} seconds.")
    return best_energy, best_solutions

if __name__ == "__main__":
    nodes, edges = load_graph('graph.json')
    
    source = 'A'
    dest = 'F'
    penalty = 100.0
    
    print(f"Routing from {source} to {dest}...")
    builder = RoutingQUBO(nodes, edges, source, dest, penalty_weight=penalty)
    Q, offset = builder.build_qubo()
    
    best_e, sols = solve_brute_force(Q, offset, builder.num_vars)
    
    print(f"Best Energy: {best_e}")
    print(f"Found {len(sols)} optimal solution(s).")
    
    for i, sol in enumerate(sols):
        path, cost = builder.decode(sol)
        valid = (best_e < penalty)
        print(f"Sol {i+1} {'(VALID)' if valid else '(INVALID)'}")
        print(f"Path: {path}")
        print(f"Cost: {cost}")
