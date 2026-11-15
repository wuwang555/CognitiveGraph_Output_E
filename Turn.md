# 认知图输出层：功能转接与接口说明

> **文档定位**：本文档面向下游模块（G2 关系图、G3 生成图、共识图、自我反思循环）的开发者。输出层（Output Layer）本质是一个**横向外部接口**，它不产生新的层级，而是负责将“外部先验（Context）”转化为“系统内部的一条能量路径（Path）”。本文档明确规定了如何提取输出层的产物，并将其无缝转接（Plug-in）到已有的上游结构中。

---

## 0. 核心思想：数据流与转接定位

输出层的主流程是：**插入节点 → 能量演化 → 提取路径**。
它产出的数据是一个轻量的 JSON/CSV 包，包含三个核心部分：
1. **先验连边**（`output_edges.csv`）：Context 节点与 G1 中原有概念的初始连边。
2. **演化路径**（`output_result.json`）：在能量动力学下，从 Context 节点流出的最省力路径。
3. **遍历记录**（可选的 `traversal_history` 增广）：若下游需要，可记录输出节点参与演化的历史。

**转接的核心动作**：把这三个数据结构，转化为下游模块的输入格式。

---

## 1. 代码结构与关键产物

### 1.1 目录结构
```text
src/output_layer/
├── config_output.py         # 参数控制（含全连接阈值、动态预算配置）
├── output_node.py           # 构建 Context 的相似度与边权
├── insert_node.py           # 插入逻辑与迭代计划
├── extract_output.py        # 三种路径提取算法（min_energy, longest_path, traversal）
├── run_with_output.py       # 主流程与数据落盘
└── run_output.py            # CLI 入口
```

### 1.2 核心产物文件
运行完成后，在 `results/output_layer/scenario_<timestamp>/` 下会生成：
- **`output_result.json`**：包含 `path`（节点序列）、`edges`（路径边权）、`total_energy`。
- **`output_edges.csv`**：包含 `concept`（相邻概念）、`energy`（初始连边能量）、`similarity`。
- **`output_node_info.json`**：包含 `n_edges_added` 和 `similarity_range`（可用于判断先验强度）。

---

## 2. 转接方案 A：接入 G2（关系图 / 不动点提取）

**目标**：让输出层产生的“先验连接”和“路径节点”参与 G2 的共现统计，进而参与不动点（Fixpoint）检测，从而看看 Context 会引发哪些概念组合。

### 2.1 输入映射
- **输出层数据**：`output_edges.csv`（Context 与 51 个概念的连边） + `output_result.json`（path）。
- **G2 所需数据**：`raw_compression_events.csv`（格式：`center, related_nodes`）。

### 2.2 具体设计（两种粒度的转接）

**粒度 1：将路径视为一次“伪压缩事件”**
如果输出路径包含多个概念（如 `__OUTPUT__ -> 队列 -> 分治算法`），可以将这条路径作为一个 `related_nodes` 集合，注入到 G2 的统计中。
```python
import json
import pandas as pd

# 1. 读取输出结果
with open('results/output_layer/scenario_xxx/output_result.json', 'r') as f:
    output_res = json.load(f)

# 2. 构造 G2 事件
path_nodes = [n for n in output_res['path'] if n != '__OUTPUT__']
if len(path_nodes) >= 2:
    # 构造一条事件记录，中心节点可以设为路径的首个真实概念
    event = {
        'center': path_nodes[0],
        'related_nodes': path_nodes[1:], 
        'source': 'output_layer'
    }
    # 3. 追加到 G2 的原始数据源
    # df = pd.read_csv('results/compression_graph/raw_compression_events.csv')
    # df = pd.concat([df, pd.DataFrame([event])])
    # df.to_csv('results/compression_graph/raw_compression_events.csv', index=False)
```

**粒度 2：将输出节点的“全连接边”作为 G2 的初始化背景**
利用 `output_edges.csv`，将 `__OUTPUT__` 节点作为一个“超级中心”，与 51 个概念的连边作为高频共现记录，喂给 G2。这能测试“特定语义先验”是否会打破原有的不动点结构。

---

## 3. 转接方案 B：接入 G3（生成图 / 内心联想）

**目标**：G3 以不动点为节点、以四维距离为边权。我们需要把输出路径“翻译”成不动点之间的迁移（Migration）或压缩（Compression），从而在 L3 层实现真正的“内心联想”。

### 3.1 输入映射
- **输出层数据**：`output_result.json`（path 概念序列）。
- **已有数据**：不动点列表（`fixpoints.json`）、四维坐标（`scale{N}.csv`）。
- **G3 所需数据**：不动点之间的四维距离 \(d_{ij}\)。

### 3.2 具体设计（映射与重计算）

**Step 1：路径概念 → 不动点映射**
输出路径里的概念（如“队列”“分治算法”）并不直接是 G3 的节点。需要先查询它属于哪个不动点。
```python
# 伪代码
path_concepts = output_res['path'][1:] # 去掉 __OUTPUT__
path_fixpoints = []
for concept in path_concepts:
    for fp in fixpoints:
        if concept in fp['core']:
            path_fixpoints.append(fp['id'])
            break
```

**Step 2：生成图上的“迁移”**
如果 `path_fixpoints` 列表中有多个不动点，它们在生成图（G3）上就会构成一条路径。你可以直接把这条路径作为 G3 的一个“强迁移”候选（`migration`），因为它是通过纯能量外推（Context）走出来的。
```python
# 伪代码：在 G3 上验证这条路径的能量改善
if len(path_fixpoints) >= 3:
    a, z, b = path_fixpoints[0], path_fixpoints[1], path_fixpoints[2]
    # 查询四维坐标，计算 d_az + d_zb vs d_ab
    # 如果 d_az + d_zb < d_ab，说明这个通过 Context 导出的迁移在四维空间也是成立的！
```

**Step 3：生成图上的“压缩”**
如果输出路径在某个不动点内部发生了多次激活（或者 `__OUTPUT__` 的连接全都指向某个不动点的成员），这就可以在 G3 中被记录为一次由外部 Context 触发的“概念压缩”候选。

---

## 4. 转接方案 C：接入共识图（社会层）

**目标**：共识图收集 \(N\) 个 G1 的 `traversal_history`。输出层可以让 \(N\) 个不同的个体输入 \(N\) 个不同的 Context，从而观察“不同外部问题在能量系统中形成的路径共识”。

### 4.1 输入映射
- **输出层数据**：`output_result.json`（path 序列）。
- **共识图所需数据**：`traversal_history`（格式：`[(u, v), (v, w), ...]`）。

### 4.2 具体设计（直接注入）

在 `run_consensus.py` 的 `collect_paths_one_round` 环节中，除了收集原系统的 `u.traversal_history`，额外将输出层的路径 append 进去。
```python
# 伪代码：在共识图的主流程中接入
for i in range(n_individuals):
    u = CognitiveUniverseEnhanced(...)
    u.initialize_semantic_network()
    
    # --- 新增：输出层介入 ---
    # 每个个体输入不同的 context
    insert_output_node_into_universe(u, context=f"用户提问{i}")
    u.evolve(iterations=interval)
    
    # 1. 收集原生路径
    all_paths.extend(u.traversal_history)
    
    # 2. 提取并收集输出路径
    output_res = extract_output(u.G, OUTPUT_NODE_NAME, config)
    if len(output_res['path']) > 1:
        # 转化为共识图需要的遍历格式 (path, traversal_type, iteration)
        all_paths.append((output_res['path'], 'output_interface', u.iteration_count))

# 更新共识图
update_consensus_graph(G, all_paths, n, sim_fn, beta)
```
**物理意义**：如果 10 个个体问了不同的 Context，但最后输出路径都经过了某一条边（比如 `队列 -> 分治算法`），这条边在共识图中的 H 会急剧增加，E 会急剧降低，从而被识别为社会级“强共识”。

---

## 5. 转接方案 D：接入自我反思循环（Self-Reflection）

**目标**：自我反思循环通过从 G3 抽象规则 K 并反馈给 G1。输出层的路径，恰恰提供了一个“来自外部输入的结构断言”，可以直接作为反馈规则。

### 5.1 输入映射
- **输出层数据**：`output_result.json`（path 序列）。
- **反思循环所需数据**：`feedback_K.json`（格式：`{edge_boosts, preferred_paths}`）。

### 5.2 具体设计（作为 K 的一部分）
如果系统的目标是“强化对某类 Context 的输出能力”，可以将该输出路径设为 `preferred_path`，并对其经过的边给予 `edge_boost`。

```python
# 伪代码：在 feedback_rules.py 中构建 K 时
K = {
    'edge_boosts': {},
    'preferred_paths': []
}

# 从输出层加载路径
with open('results/output_layer/scenario_xxx/output_result.json', 'r') as f:
    output_res = json.load(f)

path = output_res['path']
# 给予这个路径最高的遍历偏好
K['preferred_paths'].append({
    'path': path,
    'weight_multiplier': 1.5, # 强行加速这条路径的能量衰减
    'source': 'output_context'
})

# 反馈给下一轮的 G1（通过 patch 能量公式和遍历偏好）
apply_feedback_rules_to_G1(universe, K)
```
**物理意义**：这相当于系统在自我反思中，**“把上一次外部输入的答案，变成了下一次思考的思维定势”**。

---

## 6. 总结：模块分工与接口契约

通过这份说明，我们可以清晰地界定各模块的权责：

| 层级/模块 | 角色 | 核心动作 | 与输出层的接口 |
| :--- | :--- | :--- | :--- |
| **G1 (认知图)** | 基底/动力学 | 纯能量衰减、随机遍历 | 接收输出节点，跑完 N 步。 |
| **输出层 (Output Layer)** | **横向外部接口** | **将先验转为路径** | **提供 `output_result.json`。** |
| **G2 (关系图/不动点)** | 观察层/抓重点 | 统计共现，提取不动点 | 接收 `output_edges.csv` 作为初始化事件。 |
| **G3 (生成图/联想)** | 高层联想层 | 在四维距离上重算路径 | 接收 `output_result.json`，将概念映射为不动点并验证。 |
| **共识图 (社会层)** | 多体汇聚层 | 统计多人走过的边，算 `H` 值 | 接收 `output_result.json` 的 path，并入 `traversal_history`。 |
| **自我反思 (反馈层)** | 闭环记忆层 | 修改 G1 的边权偏好 | 接收 `output_result.json`，转化为 `preferred_paths`。 |

**输出层不产生概念，不修改物理法则**。它只是一个轻量级的“适配器”，把一段外部的 `Context`，翻译成基底系统能够听懂并执行的“能量微扰”，然后把系统在能量最小化驱动下的必然产物（一条路径），忠实地交还给上层建筑去解读。