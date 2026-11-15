# agent.md — 输出层需求（Output Layer）

## 0. 背景与目标

### 0.1 背景

现有项目（G1 → 共识图 → 逆向求解讨论）已具备**内化**能力：系统能吸收概念、构建能量图、演化、产生压缩/迁移、形成共现结构。但它**没有"输出"**——所有边都是"内部 → 内部"，系统从不向外吐出任何东西。

前序讨论的结论是：**输出不是缺失的机制，是缺失的节点类型**。在 G1 里加一个"输出节点"（本质是一个普通概念节点，定义内容为上下文），从它出发找能量最小的边，就是输出的最小可行形式。

### 0.2 目标

构建一个**输出层模块**，在**不修改原论文代码**的前提下：

1. 在 G1 上任意时刻插入一个"输出节点"（定义内容为上下文）；
2. 插入后基于定义计算相似度、连边、继续迭代；
3. 从输出节点出发，按**三种可选模式**提取输出：最小能量邻居 / 受能量约束的最长路径 / 受能量约束的软硬遍历；
4. 复用原论文的压缩/迁移检测，过滤出**包含输出节点**的事件；
5. 所有参数通过 `config_output.py` 集中配置。

### 0.3 边界（必须遵守）

- **不引入新公设**：输出节点就是普通概念节点，边权遵循原论文公式。
- **不判断"意义"**：输出 = 能量最小的产物。"语义正确"不在本模块范围内——能量最小就是它的意义。
- **不修改原论文文件**：`universe.py`、`detector_fixed.py`、`semantic_network.py` 等全部只读（除非十分必要）。
- **输出节点是设定**：它的"存在"和"上下文定义"由外部给定；它与已有概念的边权是内化产物。
- **扩展接口保留但不默认启用**：最长路径、软硬遍历、压缩/迁移检测都是可选配置。

---

## 1. 核心设计

### 1.1 三层结构

```
【插入层】在 G1 的某个迭代点，添加输出节点 + 基于上下文算相似度 + 连边
    ↓
【内化层】插入后继续跑（默认再跑 10000 次，受 max_iterations 约束）
    ↓
【输出层】从输出节点出发，按配置模式提取输出（最小边 / 最长路径 / 软硬遍历）
    ↓
【检测层】可选：复用原论文检测器，过滤包含输出节点的压缩/迁移
```

### 1.2 迭代计划

用户指定的迭代逻辑：

| 场景 | 总迭代次数 |
|---|---|
| 不插入（`insertion_enabled=False`） | `base_iterations`（默认 10000） |
| 插入在 `T`，无 `max_iterations` | `T + post_insertion_iterations`（默认 `T + 10000`） |
| 插入在 `T`，有 `max_iterations=M` | `min(T + post_insertion_iterations, M)` |

**示例**：
- 不插入 → 跑到 10000
- `T=4000`，无 max → 跑到 14000
- `T=4000`，`M=11000` → 跑到 11000（受限于 max）

**实现要点**：`CognitiveUniverse.evolve` 每次调用从 `iteration_count` 继续，所以可以分两次调用：先跑到 `T`，插入，再跑到 `总迭代`。

### 1.3 输出节点定义

- **节点名**：`__OUTPUT__`（常量，避免与概念名冲突）
- **定义内容**：`context`（一段文本，例如"我想说一句关于电脑的话"）
- **边权初始化**：与原论文一致，`energy = 2.0 - similarity * 1.5`，截断到 `[0.3, 2.0]`
- **相似度计算**：复用 `SemanticConceptNetwork.calculate_semantic_similarity`，需要把已有概念 + 输出节点的定义都注册进一个 `SemanticConceptNetwork` 实例

### 1.4 三种输出模式

| 模式 | 行为 | 复杂度 |
|---|---|---|
| `min_energy` | 从输出节点出发，找边权最小的邻居，返回单个节点 | O(deg) |
| `longest_path` | 受 `energy_budget` 约束，找节点数最多的路径 | O(deg^steps)（小规模可行） |
| `traversal` | 从输出节点出发，按 `hard`/`soft`/`mixed` 遍历 `traversal_steps` 步，累计边权 ≤ `energy_budget` | O(steps) |

**"能量预算"的统一语义**：`energy_budget` 是路径上累计边权（`Σ E`）的上限。`min_energy` 模式忽略它（因为只取一条边）。

### 1.5 软硬遍历的定义

沿用原论文（`BaseCognitiveGraph` 的定义，`CognitiveUniverse` 里也有）：

- **Hard traversal**：优先选能量最小的邻居（确定性高，低能耗）
- **Soft traversal**：随机选邻居（探索性高）
- **Mixed**：按 `hard_ratio` 概率走 hard，否则走 soft

**实现位置**：`extract_output.py` 里独立实现 `_find_hard_step` / `_find_soft_step`，因为原论文的 `_random_traversal` 是 `CognitiveUniverse` 的私有方法且不接收"起点节点"参数。

---

## 2. 配置入口（`config_output.py`）

**这是另一个 AI / 开发者的第一入口**。所有参数集中于此，分组清晰。

```python
"""输出层配置。所有可调参数集中于此。"""

# 输出节点名（避免与概念名冲突）
OUTPUT_NODE_NAME = "__OUTPUT__"

DEFAULT_OUTPUT_CONFIG = {
    # ---------- 基础 ----------
    'concept_config': 'A',            # A / B / C
    'num_concepts': 51,               # 概念集前 N 个
    'n_individuals': 1,               # 输出层一般单个体（保留多体接口）
    'base_iterations': 10000,         # G1 默认迭代次数
    'random_seed': 42,

    # ---------- 插入 ----------
    'insertion_enabled': True,        # 是否插入输出节点
    'insertion_iteration': None,      # None = 不插入（跑到 base_iterations）
                                      # int  = 在该迭代点插入（0 <= T <= base_iterations）
    'context': None,                  # 上下文定义（字符串）；None = 用默认占位
    'post_insertion_iterations': 10000,  # 插入后额外迭代数
    'max_iterations': None,           # 总迭代上限；None = 不限制

    # ---------- 输出提取 ----------
    'output_mode': 'min_energy',      # 'min_energy' | 'longest_path' | 'traversal'
    'energy_budget': 5.0,             # 路径累计边权上限（min_energy 忽略）
    'max_path_length': 5,             # 路径最大节点数（longest_path / traversal 用）
    'traversal_type': 'hard',         # 'hard' | 'soft' | 'mixed'
    'hard_ratio': 0.6,                # mixed 模式中 hard 的比例
    'traversal_steps': 3,             # traversal 模式步数

    # ---------- 检测扩展 ----------
    'detect_compression': True,       # 是否检测"包含输出节点"的压缩
    'detect_migration': True,         # 是否检测"包含输出节点"的迁移
    'compression_synergy': 0.76,      # 复用原论文默认值
    'migration_efficiency': 0.35,

    # ---------- 输出 ----------
    'output_root': 'results/output_layer',
}
```

**每个字段的含义已在注释中说明**。开发者只需修改这里，不需要动其他文件。

---

## 3. 代码结构

```
src/output_layer/
├── __init__.py
├── config_output.py          # 【配置入口】所有参数
├── output_node.py            # 【输出节点】创建与相似度计算
├── insert_node.py            # 【插入】在 G1 中插入输出节点 + 迭代计划计算
├── run_with_output.py        # 【主流程】跑 G1 → 插入 → 继续 → 提取 → 检测
├── extract_output.py         # 【输出提取】三种模式
├── detect_output_events.py   # 【检测】复用原论文检测器，过滤含输出节点的事件
└── run_output.py             # 【入口】命令行 / 直接运行
```

下面逐文件说明**需要写什么代码**。

---

### 3.1 `src/output_layer/__init__.py`

```python
"""输出层模块：在 G1 上插入输出节点，从它出发提取能量最小的输出。"""
```

---

### 3.2 `src/output_layer/config_output.py`

**作用**：配置入口。内容如 §2 所示。

**开发者需要提供**：常量 `OUTPUT_NODE_NAME` + 字典 `DEFAULT_OUTPUT_CONFIG`。

---

### 3.3 `src/output_layer/output_node.py`

**作用**：创建输出节点的数据结构，计算它与已有概念的相似度。

**需要实现的函数**：

```python
def build_context_similarity(
    context: str,
    existing_concepts: List[str],
    concept_config: str = "A",
) -> Dict[str, float]:
    """计算 context 与每个已有概念的语义相似度。

    实现：
        1. 构造 SemanticConceptNetwork
        2. 把 existing_concepts 的定义注册进去（从 config.CORE_CONCEPT_DEFINITIONS 读）
        3. 把 OUTPUT_NODE_NAME 的定义 = context 注册进去
        4. 对每个 concept 调 calculate_semantic_similarity(OUTPUT_NODE_NAME, concept)
    返回：{concept: similarity}
    """

def build_output_node_edges(
    similarities: Dict[str, float],
    similarity_threshold: float = 0.1,
) -> List[Tuple[str, float]]:
    """把相似度转成边。

    边权公式（与原论文一致）：
        energy = 2.0 - similarity * 1.5
        energy = max(0.3, min(2.0, energy))

    返回：[(concept, energy), ...]
    只保留 similarity > similarity_threshold 的。
    """
```

**为什么单独成文件**：相似度计算是"插入"的前置步骤，但逻辑独立（可单测），且未来若要换相似度算法（比如用 embedding），只改这一个文件。

---

### 3.4 `src/output_layer/insert_node.py`

**作用**：把输出节点真正插进 G1，并计算迭代计划。

**需要实现的函数**：

```python
def compute_iteration_plan(
    base_iterations: int,
    insertion_enabled: bool,
    insertion_iteration: Optional[int],
    post_insertion_iterations: int,
    max_iterations: Optional[int],
) -> Dict[str, int]:
    """计算迭代计划。

    返回：
        {
            'pre_insertion': int,   # 插入前的迭代数
            'post_insertion': int,  # 插入后的迭代数
            'total': int,           # 实际总迭代数
            'will_insert': bool,    # 是否真的插入
        }

    规则（见 §1.2）：
        - 不插入：pre=base, post=0, total=base
        - 插入在 T：pre=T, post=post_insertion_iterations, total=T+post
        - 有 max：total=min(T+post, max)
        - T 超出 [0, base] 时截断并警告
    """

def insert_output_node_into_universe(
    universe: Any,                       # CognitiveUniverse 实例
    context: str,
    concept_config: str,
    similarity_threshold: float = 0.1,
) -> Dict[str, Any]:
    """在 universe.G 上插入输出节点。

    步骤：
        1. existing = list(universe.G.nodes())
        2. sims = build_context_similarity(context, existing, concept_config)
        3. edges = build_output_node_edges(sims, similarity_threshold)
        4. universe.G.add_node(OUTPUT_NODE_NAME, is_output=True,
                               original_weight=2.0, similarity=...)
        5. for concept, energy in edges:
               universe.G.add_edge(OUTPUT_NODE_NAME, concept,
                                   weight=energy,
                                   original_weight=energy,
                                   similarity=sims[concept],
                                   traversal_count=0)
               universe.last_activation_time[(OUTPUT_NODE_NAME, concept)] = 0

    返回：
        {
            'output_node': OUTPUT_NODE_NAME,
            'context': context,
            'n_edges_added': int,
            'edges': [(concept, energy), ...],
        }
    """
```

**关键约束**：插入节点后，`universe.G` 是一个**无向图**（原论文的 G1 是无向的）。输出节点与所有概念连**无向边**。这与 G1 一致。

---

### 3.5 `src/output_layer/run_with_output.py`

**作用**：主流程——跑 G1、插入、继续跑、调用提取、调用检测。

**需要实现的函数**：

```python
def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """完整流程。

    步骤：
        1. 根据 concept_config 设置环境变量 CONFIG_METHOD
        2. 创建 CognitiveUniverse（原论文的，不是 Enhanced）
        3. initialize_semantic_network()
        4. plan = compute_iteration_plan(...)
        5. if plan['pre_insertion'] > 0:
               universe.evolve(iterations=plan['pre_insertion'],
                               observation_interval=100)
        6. if plan['will_insert']:
               insert_output_node_into_universe(universe, context, ...)
        7. if plan['post_insertion'] > 0:
               universe.evolve(iterations=plan['post_insertion'],
                               observation_interval=100)
        8. output = extract_output(universe.G, OUTPUT_NODE_NAME, config)
        9. if config['detect_compression']:
               compressions = detect_compression_with_output(...)
           if config['detect_migration']:
               migrations = detect_migration_with_output(...)
        10. 落盘所有结果（见 §5）
        11. 返回 summary 字典
    """
```

**注意事项**：
- `CognitiveUniverse.evolve` 会 `print` 大量信息，用 `contextlib.redirect_stdout` 静音（可选 `--verbose` 开关）
- `evolve` 是累加的（`iteration_count` 递增），所以分两次调用是安全的
- 插入节点后，`universe.last_activation_time` 需要为新边补时间戳

---

### 3.6 `src/output_layer/extract_output.py`

**作用**：三种输出模式的实现。

**需要实现的函数**：

```python
def extract_output(
    G: nx.Graph,
    output_node: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """根据 config['output_mode'] 分派。

    返回统一结构：
        {
            'mode': str,
            'path': [str, ...],        # 节点序列（min_energy 时长度为 2）
            'edges': [dict, ...],      # 每条边的 {u, v, weight}
            'total_energy': float,
            'energy_budget': float,
        }
    """

def find_min_energy_neighbor(
    G: nx.Graph,
    output_node: str,
) -> Dict[str, Any]:
    """从 output_node 出发，找边权最小的邻居。

    实现：
        neighbors = list(G.neighbors(output_node))
        neighbors.sort(key=lambda n: G[output_node][n]['weight'])
        best = neighbors[0]
        return {'mode': 'min_energy',
                'path': [output_node, best],
                'edges': [{'u': output_node, 'v': best,
                           'weight': G[output_node][best]['weight']}],
                'total_energy': G[output_node][best]['weight'],
                'energy_budget': None}
    """

def find_longest_path(
    G: nx.Graph,
    output_node: str,
    energy_budget: float,
    max_length: int,
) -> Dict[str, Any]:
    """在 energy_budget 约束下，找节点数最多的路径。

    实现：DFS，记录当前路径 + 累计能量。
        剪枝：如果累计能量 + 最小出边 > budget，停止扩展。
        只在"节点数更多"时更新 best（节点数相同取能量更低的）。
    注意：这是"最长路径"的近似，不是全局最优（避免指数爆炸）。
    """

def traverse_from_output(
    G: nx.Graph,
    output_node: str,
    traversal_type: str,       # 'hard' | 'soft' | 'mixed'
    hard_ratio: float,
    steps: int,
    energy_budget: float,
) -> Dict[str, Any]:
    """从 output_node 出发，按软硬遍历走 steps 步。

    实现：
        current = output_node
        path = [current]
        total = 0
        for _ in range(steps):
            neighbors = [n for n in G.neighbors(current) if n not in path]
            if not neighbors: break
            if traversal_type == 'hard':
                neighbors.sort(key=lambda n: G[current][n]['weight'])
                next_node = neighbors[0]
            elif traversal_type == 'soft':
                next_node = random.choice(neighbors)
            else:  # mixed
                if random.random() < hard_ratio:
                    neighbors.sort(key=lambda n: G[current][n]['weight'])
                    next_node = neighbors[0]
                else:
                    next_node = random.choice(neighbors)
            w = G[current][next_node]['weight']
            if total + w > energy_budget:
                break
            path.append(next_node)
            total += w
            current = next_node
        return {'mode': 'traversal', 'path': path, ...}
    """
```

**关键约束**：
- 所有模式都返回**统一结构**（`path` / `edges` / `total_energy`），便于下游处理
- `energy_budget` 在 `min_energy` 模式下被忽略（因为只取一条边）
- 遍历不重复访问节点（`n not in path`），避免死循环

---

### 3.7 `src/output_layer/detect_output_events.py`

**作用**：复用原论文 `EmergenceDetectorFixed`，过滤出"包含输出节点"的事件。

**需要实现的函数**：

```python
def detect_compression_with_output(
    G: nx.Graph,
    energy_history: List[float],
    traversal_history: List[Any],
    output_node: str,
    synergy_threshold: float = 0.76,
) -> pd.DataFrame:
    """检测包含输出节点的压缩事件。

    实现：
        from src.emergence.detector_fixed import EmergenceDetectorFixed
        detector = EmergenceDetectorFixed({
            'compression_synergy': synergy_threshold,
        })
        all_compressions = detector.detect_spontaneous_compression(
            G, energy_history, traversal_history
        )
        # 过滤：center == output_node 或 output_node in related_nodes
        filtered = [
            c for c in all_compressions
            if c['center'] == output_node
            or output_node in c.get('related_nodes', [])
        ]
        return pd.DataFrame(filtered)
    """

def detect_migration_with_output(
    G: nx.Graph,
    traversal_history: List[Any],
    current_iteration: int,
    output_node: str,
    efficiency_threshold: float = 0.35,
) -> pd.DataFrame:
    """检测包含输出节点的迁移事件。

    实现：
        detector = EmergenceDetectorFixed({
            'migration_efficiency': efficiency_threshold,
        })
        all_migrations = detector.detect_emergent_migration(
            G, traversal_history, current_iteration
        )
        filtered = [
            m for m in all_migrations
            if output_node in m.get('path', [])
        ]
        return pd.DataFrame(filtered)
    """
```

**为什么单独成文件**：检测逻辑完全复用原论文，这里只做"过滤 + 转 DataFrame"，与输出提取解耦。

---

### 3.8 `src/output_layer/run_output.py`

**作用**：入口脚本。

**需要实现**：

```python
"""输出层入口。

用法：
    # 不插入（基线）
    python -m src.output_layer.run_output --no-insert

    # 在 4000 次插入，上下文为"我想说一句关于电脑的话"
    python -m src.output_layer.run_output \
        --insert-at 4000 --context "我想说一句关于电脑的话" \
        --output-mode min_energy

    # 最长路径模式
    python -m src.output_layer.run_output \
        --insert-at 4000 --context "电脑坏了" \
        --output-mode longest_path --energy-budget 3.0 --max-path-length 5

    # 软硬遍历模式
    python -m src.output_layer.run_output \
        --insert-at 4000 --context "电脑坏了" \
        --output-mode traversal --traversal-type mixed --traversal-steps 4
"""

def parse_args() -> argparse.Namespace: ...
def main() -> None:
    args = parse_args()
    config = DEFAULT_OUTPUT_CONFIG.copy()
    # 用 CLI 参数覆盖 config
    ...
    run_pipeline(config)
```

**CLI 参数**对应 `DEFAULT_OUTPUT_CONFIG` 的字段，全部可选，默认用 config 里的值。

---

## 4. 关键算法（伪代码）

### 4.1 迭代计划

```python
def compute_iteration_plan(base, enabled, T, post, max_iter):
    if not enabled or T is None:
        return {'pre': base, 'post': 0, 'total': base, 'will_insert': False}
    T = max(0, min(T, base))
    total = T + post
    if max_iter is not None:
        total = min(total, max_iter)
    post_actual = total - T
    return {'pre': T, 'post': post_actual, 'total': total, 'will_insert': True}
```

### 4.2 插入节点

```python
def insert_output_node(G, context, concept_config):
    existing = list(G.nodes())
    sims = build_context_similarity(context, existing, concept_config)
    G.add_node(OUTPUT_NODE_NAME, is_output=True)
    for concept, sim in sims.items():
        if sim <= 0.1:
            continue
        energy = max(0.3, min(2.0, 2.0 - sim * 1.5))
        G.add_edge(OUTPUT_NODE_NAME, concept,
                   weight=energy, original_weight=energy,
                   similarity=sim, traversal_count=0)
    return {'n_edges_added': ..., 'edges': ...}
```

### 4.3 最小能量邻居

```python
def find_min_energy_neighbor(G, output_node):
    neighbors = list(G.neighbors(output_node))
    neighbors.sort(key=lambda n: G[output_node][n]['weight'])
    return neighbors[0]
```

### 4.4 最长路径（DFS + 剪枝）

```python
def find_longest_path(G, start, budget, max_len):
    best = {'path': [start], 'energy': 0.0}
    def dfs(node, path, energy):
        if len(path) > len(best['path']) or (
            len(path) == len(best['path']) and energy < best['energy']):
            best['path'] = path.copy()
            best['energy'] = energy
        if len(path) >= max_len:
            return
        for nb in G.neighbors(node):
            if nb in path: continue
            w = G[node][nb]['weight']
            if energy + w > budget: continue
            dfs(nb, path + [nb], energy + w)
    dfs(start, [start], 0.0)
    return best
```

### 4.5 软硬遍历

```python
def traverse(G, start, type, hard_ratio, steps, budget):
    path, total, current = [start], 0.0, start
    for _ in range(steps):
        nbs = [n for n in G.neighbors(current) if n not in path]
        if not nbs: break
        if type == 'hard' or (type == 'mixed' and random.random() < hard_ratio):
            nbs.sort(key=lambda n: G[current][n]['weight'])
            nxt = nbs[0]
        else:
            nxt = random.choice(nbs)
        w = G[current][nxt]['weight']
        if total + w > budget: break
        path.append(nxt); total += w; current = nxt
    return {'path': path, 'total_energy': total}
```

---

## 5. 输出结构

```
results/output_layer/
├── scenario_<timestamp>/
│   ├── iteration_plan.json          # {'pre', 'post', 'total', 'will_insert'}
│   ├── output_node_info.json        # {'output_node', 'context', 'n_edges_added'}
│   ├── output_edges.csv             # output_node 的所有边 (concept, energy, similarity)
│   ├── output_result.json           # {'mode', 'path', 'edges', 'total_energy', 'energy_budget'}
│   ├── output_compressions.csv      # 包含 output_node 的压缩事件（可为空）
│   ├── output_migrations.csv        # 包含 output_node 的迁移事件（可为空）
│   └── g1_summary.json              # {'n_nodes', 'n_edges', 'mean_weight', 'final_energy'}
└── summary.csv                      # 每次运行的汇总（一行）
```

`output_result.json` 字段：

```json
{
  "mode": "min_energy",
  "path": ["__OUTPUT__", "算法"],
  "edges": [
    {"u": "__OUTPUT__", "v": "算法", "weight": 0.42}
  ],
  "total_energy": 0.42,
  "energy_budget": null
}
```

`summary.csv` 列：

```
timestamp, concept_config, num_concepts, insertion_enabled,
insertion_iteration, total_iterations, output_mode,
n_output_edges, output_path, output_total_energy,
n_compressions, n_migrations
```

---

## 6. 验收标准

### 6.1 功能验收

| # | 验收项 | 判定 |
|---|---|---|
| 1 | 不插入时跑到 10000 次 | `iteration_plan.json` 的 `total == 10000` |
| 2 | `T=4000` 无 max 时跑到 14000 | `total == 14000` |
| 3 | `T=4000, M=11000` 时跑到 11000 | `total == 11000` |
| 4 | 插入节点的边权遵循 `2.0 - sim*1.5` | `output_edges.csv` 逐行验证 |
| 5 | `min_energy` 返回单边 | `len(path) == 2` |
| 6 | `longest_path` 累计能量 ≤ budget | `total_energy <= energy_budget` |
| 7 | `traversal` 步数 ≤ steps | `len(path) - 1 <= traversal_steps` |
| 8 | 压缩检测过滤出含输出节点的事件 | `output_compressions.csv` 每行 center 或 related_nodes 含 `__OUTPUT__` |
| 9 | 迁移检测过滤出含输出节点的路径 | `output_migrations.csv` 每行 path 含 `__OUTPUT__` |

### 6.2 边界验收

| # | 验收项 | 判定 |
|---|---|---|
| 10 | 不修改原论文文件 | `git diff src/core src/emergence src/utils` 为空 |
| 11 | 输出节点是普通概念节点 | `G.nodes["__OUTPUT__"]` 只多一个 `is_output=True` 属性 |
| 12 | 无"意义"判断 | 代码里无"语义正确"、"评估"相关逻辑 |
| 13 | 配置集中 | 所有可调参数在 `config_output.py` |

### 6.3 快速验证命令

```bash
# 基线（不插入）
python -m src.output_layer.run_output --no-insert

# 插入 + 最小能量
python -m src.output_layer.run_output \
    --insert-at 4000 --context "我想说一句关于电脑的话" \
    --output-mode min_energy

# 插入 + 最长路径
python -m src.output_layer.run_output \
    --insert-at 4000 --context "电脑坏了" \
    --output-mode longest_path --energy-budget 3.0

# 插入 + 软硬遍历
python -m src.output_layer.run_output \
    --insert-at 4000 --context "电脑坏了" \
    --output-mode traversal --traversal-type mixed --traversal-steps 4
```

---

## 7. 与原论文/现有代码的关系

### 7.1 直接复用（不改）

| 原文件 | 用到什么 |
|---|---|
| `config.py` | `CORE_CONCEPT_DEFINITIONS`、`CONFIG_METHOD` |
| `src/core/semantic_network.py` | `SemanticConceptNetwork.calculate_semantic_similarity` |
| `src/emergence/universe.py` | `CognitiveUniverse`（`evolve`、`initialize_semantic_network`、`traversal_history`、`energy_history`） |
| `src/emergence/detector_fixed.py` | `EmergenceDetectorFixed.detect_spontaneous_compression` / `detect_emergent_migration` |

### 7.2 新增（本模块）

| 文件 | 作用 |
|---|---|
| `config_output.py` | 配置入口 |
| `output_node.py` | 相似度计算 + 边构造 |
| `insert_node.py` | 迭代计划 + 插入逻辑 |
| `run_with_output.py` | 主流程 |
| `extract_output.py` | 三种输出模式 |
| `detect_output_events.py` | 检测 + 过滤 |
| `run_output.py` | CLI 入口 |

---

## 8. 给下一个 AI / 开发者的提示

1. **先读 `config_output.py`**，它是所有参数的入口。
2. **先跑 `--no-insert` 基线**，确认 G1 能正常跑 10000 次。
3. **再跑 `--insert-at 4000`**，确认输出节点被插入、边权正确。
4. **最后试三种输出模式**，确认 `output_result.json` 结构一致。
5. **不要"优化"成 embedding 相似度**——当前用 `SemanticConceptNetwork`，与原论文一致，是设计选择，不是遗漏。
6. **不要加"意义判断"**——能量最小就是输出，语义正确不在范围内。
7. **不要改原论文文件**——所有改动在 `src/output_layer/` 内。
8. **`CognitiveUniverse.evolve` 是累加的**——分两次调用（插入前、插入后）是安全的，不需要手动重置 `iteration_count`。
9. **插入节点后记得补 `last_activation_time`**——否则遗忘机制会把它当"从未激活"处理。
10. **`compute_iteration_plan` 的边界要测**——`T=0`、`T=10000`、`M<T+post` 三种情况都要覆盖。

---

## 9. 一句话总结

**输出层 = 在 G1 上插一个"上下文节点"，跑完之后从它出发找能量最小的边（或受能量约束的最长路径 / 软硬遍历）。它复用原论文的 `CognitiveUniverse` 和 `EmergenceDetectorFixed`，不引入新机制，不判断"意义"，所有参数集中在 `config_output.py`。七个文件分工明确：配置、节点、插入、主流程、提取、检测、入口。验收标准是"迭代计划正确、边权公式正确、三种模式都能出结果、检测能过滤出含输出节点的事件"。**