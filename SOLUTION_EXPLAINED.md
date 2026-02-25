# How the Solver Works — and Why It Needs More Than Just SA

## The goal

Find the shortest cable route from node **A** to node **F** through a graph of ducts.
We encode this as a **QUBO** (binary optimisation problem) and solve it with
**myQLM's Simulated Annealing (SA)** QPU.

---

## Step 1 — Build the QUBO

Each undirected duct becomes **two directed binary variables**, e.g. edge A–B becomes:
- `x(A→B)` = 1 if the cable travels A to B, 0 otherwise
- `x(B→A)` = 1 if the cable travels B to A, 0 otherwise

The QUBO objective is a matrix `M` where:
- **Diagonal terms** = edge costs (cheap edges are rewarded)
- **Off-diagonal penalty terms** = "flow conservation" constraints

Flow conservation means: at every intermediate node, the number of cables entering
must equal the number leaving. At the **source** one more cable leaves than enters;
at the **destination** one more cable enters than leaves.

Any assignment of 0s and 1s that violates this rule gets a large **penalty P** added
to its energy. Valid paths have energy ≈ path cost (~4–8 m). Invalid assignments
have energy ≥ P (~15).

---

## Step 2 — Tell myQLM about the problem

myQLM's `QUBO` class minimises:

```
H = -x^T Q x - E_Q
```

Our objective is `x^T M x + offset`, so we pass **`Q = -M`** and **`offset_q = -offset`**.
This sign flip is the only "translation" needed between our formulation and myQLM's convention.

---

## Step 3 — Run SA

```python
qpu = SimulatedAnnealing(temp_t=temp_t, n_steps=5000, seed=seed)
result = qpu.submit(job)
```

SA starts from a random spin configuration, then repeatedly flips one spin at a time.
Flips that lower energy are always accepted. Flips that raise energy are accepted with
probability `exp(-ΔE / T)` — more likely at high temperature, nearly impossible at low.
Temperature cools from `T_max=50` down to `T_min=0.1` over 5 000 steps.

---

## Why not just run SA and read the answer?

Two problems appeared in practice.

### Problem 1 — SA returns the *final* state, not the *best* state

SA is a random walk. As it cools, it moves less and less. Wherever it *stops* is what
gets returned — not the lowest-energy point it *visited* along the way.

**Example:** SA might reach energy 4.6 (valid path) at step 3 000, then wander away
to energy 38.9 by step 5 000. It returns 38.9.

The fix is a **greedy local descent** after every SA run:

```
while there exists a single flip that lowers energy:
    make the best such flip
```

This is cheap (only 16 variables) and deterministic. Starting from 38.9, it
systematically removes "wrong" edges until the energy drops to 4.6 — a valid path.

### Problem 2 — SA can produce loops (A→B→E→B→E→…)

The QUBO only enforces flow conservation per node. It does not forbid a cable going
**B→E and E→B simultaneously** — both being active still satisfies conservation
(B sends one cable out via E, and receives one cable in from E). The energy for such
a "loop" can be lower than many invalid states, so SA happily lands there.

**Example:** `{A→B, B→E, E→B, B→C, C→F}` satisfies all constraints and has energy 5.7,
but the cable loops B–E–B for no reason.

The fix is a **cycle-removal pass**:

```
for every directed edge pair (u→v) and (v→u):
    if both are active, set both to 0
```

Removing a back-and-forth pair never breaks flow conservation (the net flow at each
node is unchanged) and always reduces cost. After removal a second greedy descent
finds the nearest clean minimum.

---

## The full pipeline per restart

```
Random start (SA)
      ↓
myQLM SimulatedAnnealing  (global exploration)
      ↓
Greedy local descent       (fix "final ≠ best" issue)
      ↓
Cycle removal              (remove back-and-forth pairs)
      ↓
Greedy local descent again (tidy up after removal)
      ↓
Check: is this a valid simple path from A to F?
```

This runs 30 times with different random seeds. The best valid result across all
restarts is reported.

---

## Why penalty P = 15?

Any **invalid** state has at least one constraint violation, adding ≥ P to its energy.
The longest valid simple path in this graph costs about 8.3 m.

As long as **P > 8.3**, every valid path (energy ≤ 8.3) is cheaper than every invalid
state (energy ≥ P = 15). The QUBO global minimum is therefore always a valid path.

Setting P too large (e.g. 100) makes the energy barriers huge (~100 units), so SA
needs an extremely high temperature to cross them — it gets stuck. P = 15 keeps
barriers small (≤ 15 units) while still guaranteeing correctness.

---

## Result

| Restart | SA energy | After pipeline | Valid? | Path |
|---------|-----------|----------------|--------|------|
| 1  | 38.9 | **4.6** | ✅ | A→B→E→F |
| 2  | 39.5 | **5.5** | ✅ | A→D→E→B→C→F |
| …  | …    | …       | …  | … |
| **Best** | — | **4.3 m** | ✅ | **A→D→F** |

30 out of 30 restarts produce a valid path. The best found is **A→D→F** at
**4.3 m** — matching the brute-force optimal exactly.

---

## Problem 3 — Greedy descent gets stuck in the wrong local minimum

Even with the two fixes above, the solver was consistently returning **A→B→E→F**
(4.6 m) instead of the global optimum **A→D→F** (4.3 m). 144 out of 200 SA runs
already contained both `A→D` and `D→F` in the active set — so the raw material
for the correct answer was there. Yet greedy descent always threw it away.

### Why?

Consider a typical SA output: `{A→B, B→C, C→F, D→F, A→D}`.

Node A has *two* outgoing edges (`A→B` and `A→D`), which violates the flow
constraint and adds a penalty of 15. Greedy descent must remove one of them.
It always picks the **single best flip** — i.e. the removal that saves the most
energy in that one step:

| Remove edge | Energy saved |
|-------------|-------------|
| `D→F`       | **−4.1** (edge cost) |
| `A→B`       | −2.0 (edge cost) |

Greedy descent removes `D→F` because 4.1 > 2.0. But `D→F` is the only link
connecting D to F, so the A→D→F path is destroyed. The solver is left with
`{A→B, B→C, C→F}` — a valid path, but costing 5.1 m.

The cheaper path A→D→F *requires* removing `A→B` first, even though it saves
less energy in that single step. Deterministic greedy descent can never make
that choice.

### The fix — randomised greedy descent

Instead of always picking the *best* flip, each greedy descent trial picks a
*random* improving flip at every step. The descent is run **30 times** from the
same SA starting point; the best result across all trials is kept.

```
for each of 30 random orderings:
    while there is any flip that lowers energy:
        pick one at random and apply it
keep the result with the lowest final energy
```

With random ordering, roughly 1 in 3 trials will remove `A→B` before `D→F`
at the critical step. That run then follows the chain:

```
remove A→B  →  {B→C, C→F, D→F, A→D}  (B now dangling)
remove B→C  →  {C→F, D→F, A→D}        (C now dangling)
remove C→F  →  {D→F, A→D}             = A→D→F  ✅  energy 4.3
```

The best-of-30 over 30 SA restarts reliably finds the global optimum.

### Updated pipeline

```
Random start (SA)
      ↓
myQLM SimulatedAnnealing       (global exploration)
      ↓
Multi-start randomised descent (escape deterministic local minima)
      ↓
Cycle removal                  (remove back-and-forth pairs)
      ↓
Multi-start randomised descent (tidy up after removal)
      ↓
Check: is this a valid simple path from A to F?
```
