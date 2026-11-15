# Cognitive Graph Output Layer: A Minimum Viable Output Mechanism Based on Energy Dynamics

> This document describes the **output layer module** built on the foundation of *The Geometry, Algebra, and Dynamics of Cognition* (hereinafter referred to as "the original paper").
> **The tone is consistent with the original paper: do not pursue fitting, only pursue computability; do not judge "meaning," only output the product of "minimum energy."**

---

## 0. Positioning: The "External Interface" of the Complex-System Foundation

Through two postulates (cognitive spacetime + energy minimization), the original paper uses extremely minimal code to construct a **complex-system foundation** that can spontaneously exhibit power-law decay (\(R^2>0.99\)), Zipf distribution (slope about -1.35), and cross-scale stability. With an extremely restrained attitude, the original paper only observes its internal phenomena (compression and migration), demonstrating its reliability as a thought-experiment platform.

**This module is the "external interface" on top of the original paper's foundation.**
It does not modify the first-order dynamics of the original system; it only answers one extremely simple question:
> **When an external prior (Context) is input, where does it naturally flow in this energy-driven complex system?**

Its essence is: on the original fully connected energy graph, insert a "context node," let it participate in evolution, and finally search from it for the path with **minimum energy**. This is the minimum viable form of interaction between a complex system and the external world.

---

## 1. Core Mechanism: Take a Look, Think for a While, Give a Conclusion

The operating logic of the output layer can be precisely summarized as: **insert, evolve, extract**.

### 1.1 Insert (Prior Input)
- **Action**: In the G1 graph (at the current iteration count \(T\)), insert an `__OUTPUT__` node.
- **Connections**: Based on the semantic similarity of the original paper, compute the similarity between `Context` and all existing concepts. **Fully connected by default** (maintaining the dense-network foundation of the complex system), with edge weights initialized according to the original formula \(E = 2.0 - 1.5 \times \text{sim}\).
- **Physical meaning**: This is "**taking a look at the problem**." The prior semantics are injected into the network at this moment.

### 1.2 Evolve (Internal Thinking)
- **Action**: Continue running for \(N\) iterations (default 10,000).
- **Mechanism**: Execute the original system's `basic_energy_optimization` (randomly select an edge and decrease its weight) and `apply_basic_forgetting` (pull back toward the initial value).
- **Physical meaning**: This is "**closing your eyes and thinking**." There is no semantic bias in the system; structure is reshaped purely by energy gradients and random walks.

### 1.3 Extract (Output Result)
- **Action**: Starting from `__OUTPUT__`, search for a path according to the specified mode (`min_energy` / `longest_path` / `traversal`).
- **Physical meaning**: Output = the **only physical fact** in the system's current state: the route to the lowest-energy neighbor.

---

## 2. Dynamic Energy Budget: An "Output Cap" Tied to Evolution

To avoid arbitrarily deciding "how far a path can go," this module introduces a **Dynamic Energy Budget**.

\[
Budget = E_{insert} - E_{final}
\]

- \(E_{insert}\): the global total energy when the output node is inserted (including the output node's edges).
- \(E_{final}\): the global total energy at the end of evolution.
- **Physical meaning**: The energy "saved" by the system during \(N\) steps of thinking is the energy this output "can spend."
  - If the system saves no energy (\(Budget \le 0\)), the output is truncated.
  - If the system saves a great deal of energy (\(Budget > 0\)), the system can take a very long path (equivalent to thinking very deeply).

Two measurement modes are provided:
1. **`total` (total energy)**: the budget is very large (often several hundred); in this case, path length is constrained by `max_path_length`, not by energy.
2. **`mean` (mean energy)**: the budget is very small (about 0.9); in this case, the path is **truly truncated by local energy minima**.

---

## 3. Three Output Modes

Based on the original paper's concepts of "hard/soft traversal," three extraction strategies are provided:

1. **`min_energy` (minimal output)**
   Take only one edge: `__OUTPUT__ -> lowest-energy neighbor`. Does not consume the dynamic budget.
2. **`longest_path` (deep association)**
   Use Beam Search (to prevent DFS exponential explosion) to find, under the `energy_budget` constraint, the path with the largest number of nodes.
3. **`traversal` (soft-hard hybrid)**
   Step according to `hard` (choose the smallest edge) / `soft` (choose an edge randomly) / `mixed` (mix by ratio) strategies, under the constraints of `max_path_length` and `energy_budget`.

---

## 4. Real Runs and Phenomenon Analysis: Why Is the Output "Animal-Level"?

In a real run, we input a Context about "complexity analysis" (containing strong semantic words such as "time complexity") and obtained the following observations:

| Parameter | Value |
| :--- | :--- |
| Output node edges | 51 / 51 (fully connected) |
| Sim range | 0.0600 ~ 0.3600 |
| Concept with highest Sim | Time complexity (0.36) |
| **Final output path** | **`__OUTPUT__ -> Queue -> Divide-and-Conquer Algorithm -> Complexity Theory`** |
| Total energy of output path | 0.8025 |

### Phenomenon 1: Why Didn't the High-Semantic Concept Win?
The initial sim of "Time complexity" is 0.36, with edge weight 1.46; the sim of "Queue" is 0.06, with edge weight 1.91.
However, after 10,000 completely unbiased random walks (Poisson activation, mean 43 times), **as long as "Queue" is stepped on 3 more times than "Time complexity" (\(0.9^3 \approx 0.729\)), the edge weights reverse**.

This precisely proves that the original paper's foundation is a **true complex system**: it has no semantic bias, only energy dynamics. This is like "**taking a look at the problem, then closing your eyes and spacing out for 10,000 steps**"—the initial semantic prior is completely diluted by random noise and the forgetting mechanism.

### Phenomenon 2: Why "Queue"?
Because in this random walk, "Queue" happens to be the least-effort channel under this node's path dependence. This is a **physical necessity**. It faithfully reflects the energy basin of the complex system in its current state.

---

## 5. Interfaces with Upstream and Downstream Modules (Why It Is Only an "Interface")

**The output layer itself is not responsible for "concept generation" or "semantic extraction"**; these have already been completed in the original paper's extension layers. The output layer's path can directly serve as their data source:

1. **Feed to G2 (relation graph)**: The 51 edges inserted by the output layer can serve as an initialization source for the G2 co-occurrence graph, allowing subsequent fixpoint detection to "capture the essentials."
2. **Feed to the Generative Graph (L3 inner association)**: If the output path passes through several fixpoints (e.g., "Divide-and-Conquer Algorithm" belongs to a certain fixpoint), then the distance sequence of this path in the four-dimensional space is a real instance of "inner association."
3. **Feed to the Consensus Graph (social layer)**: \(N\) individuals input different Contexts and produce \(N\) paths, which are merged into the consensus graph's traversal history to compute a "social-level consensus output."

---

## 6. Boundaries and Declarations

1. **Do not modify the original system**: All code is under `src/output_layer/`; the original paper's code is modified zero times.
2. **Do not judge meaning**: The output path is not guaranteed to be "semantically correct"; it only guarantees "minimum energy." The final interpretation of semantics lies with human readers or downstream fixpoint detectors.
3. **Fully connected, maintaining the original graph's tone**: By default, the output node is connected to all concepts, maintaining the physical setting of the original system's dense graph.
4. **No working memory**: The graph is all the memory. The inserted node will not "be forgotten," but its edge weights will be diluted by the dynamics.

---

## 7. One-Sentence Summary

**The original paper uses "minimal postulates" to run through a complex system, proving that "energy minimization" can spontaneously produce power laws and Zipf; the output layer, on this minimal foundation, provides an outward interface—input a prior, and see which path it flows out along under pure energy dynamics. It does not guarantee semantics; it only provides a faithful physical channel determined by energy gradients.**