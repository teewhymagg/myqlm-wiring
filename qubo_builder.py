import numpy as np

class RoutingQUBO:
    def __init__(self, nodes, edges, source, dest, penalty_weight=10.0):
        self.nodes = nodes
        self.edges = edges
        self.source = source
        self.dest = dest
        self.P = penalty_weight
        self.vars = []
        self.cost_map = {}
        for e in edges:
            u, v, cost = e['u'], e['v'], e['cost']
            # forward
            self.vars.append((u, v))
            self.cost_map[(u, v)] = cost
            # backward
            self.vars.append((v, u))
            self.cost_map[(v, u)] = cost
            
        self.num_vars = len(self.vars)
        self.var_to_idx = {var: i for i, var in enumerate(self.vars)}
        self.incoming = {node: [] for node in nodes}
        self.outgoing = {node: [] for node in nodes}
        
        for u, v in self.vars:
            self.outgoing[u].append((u, v))
            self.incoming[v].append((u, v))
            
    def _add_interaction(self, Q, i, j, val):
        if i == j:
            Q[i, i] += val
        else:
            Q[i, j] += val / 2
            Q[j, i] += val / 2

    def build_qubo(self):
        Q = np.zeros((self.num_vars, self.num_vars))
        offset = 0.0
        
        for i, var in enumerate(self.vars):
            Q[i, i] += self.cost_map[var]
            
        for node in self.nodes:
            in_vars = [self.var_to_idx[var] for var in self.incoming[node]]
            out_vars = [self.var_to_idx[var] for var in self.outgoing[node]]
            
            if node == self.source:
                target = 1.0
            elif node == self.dest:
                target = -1.0
            else:
                target = 0.0
                
            offset += self.P * (target ** 2)
            
            for i in out_vars:
                self._add_interaction(Q, i, i, self.P)
                for j in out_vars:
                    if i < j:
                        self._add_interaction(Q, i, j, 2 * self.P)
                        
            for i in in_vars:
                self._add_interaction(Q, i, i, self.P)
                for j in in_vars:
                    if i < j:
                        self._add_interaction(Q, i, j, 2 * self.P)
                        
            for i in out_vars:
                for j in in_vars:
                    self._add_interaction(Q, i, j, -2 * self.P)
                    
            for i in out_vars:
                self._add_interaction(Q, i, i, -2 * target * self.P)
                
            for i in in_vars:
                self._add_interaction(Q, i, i, 2 * target * self.P)
                
        Q = (Q + Q.T) / 2
        return Q, offset

    def decode(self, solution):
        path_edges = []
        cost = 0.0
        
        chosen = []
        for i, val in enumerate(solution):
            if val == 1:
                var = self.vars[i]
                chosen.append(var)
                cost += self.cost_map[var]
                
        if not chosen:
            return [], 0.0
            
        next_node = {}
        for u, v in chosen:
            next_node[u] = v
            
        current = self.source
        sorted_path = []
        
        max_steps = len(chosen) 
        
        while current in next_node and max_steps > 0:
            nxt = next_node[current]
            sorted_path.append((current, nxt))
            current = nxt
            max_steps -= 1
            if current == self.dest:
                break
                
        used_edges = set(sorted_path)
        for e in chosen:
            if e not in used_edges:
                sorted_path.append(e)
                
        return sorted_path, cost
