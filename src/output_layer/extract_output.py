# -*- coding: utf-8 -*-
"""三种输出模式：min_energy / longest_path / traversal。

所有模式都返回统一结构：
    {
        'mode': str,
        'path': List[str],
        'edges': List[Dict],   # 每项 {u, v, weight}
        'total_energy': float,
        'energy_budget': float | None,
    }
"""

import random
from typing import Any, Dict, List

import networkx as nx


def extract_output(G: nx.Graph, output_node: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """根据 config['output_mode'] 分派。"""
    mode = config.get('output_mode', 'min_energy')
    if mode == 'min_energy':
        return find_min_energy_neighbor(G, output_node)
    elif mode == 'longest_path':
        return find_longest_path(
            G, output_node,
            config.get('energy_budget', 5.0),
            config.get('max_path_length', 5),
            config.get('beam_width', 100),
        )
    elif mode == 'traversal':
        return traverse_from_output(
            G, output_node,
            config.get('traversal_type', 'hard'),
            config.get('hard_ratio', 0.6),
            config.get('traversal_steps', 3),
            config.get('energy_budget', 5.0),
        )
    else:
        raise ValueError(f"Unknown output_mode: {mode}")


def _empty_result(mode: str, output_node: str, energy_budget=None):
    return {
        'mode': mode,
        'path': [output_node] if output_node else [],
        'edges': [],
        'total_energy': 0.0,
        'energy_budget': energy_budget,
    }


def find_min_energy_neighbor(G: nx.Graph, output_node: str) -> Dict[str, Any]:
    """从 output_node 出发，找边权最小的邻居。"""
    if output_node not in G:
        return _empty_result('min_energy', output_node)

    neighbors = list(G.neighbors(output_node))
    if not neighbors:
        return _empty_result('min_energy', output_node)

    neighbors.sort(key=lambda n: G[output_node][n]['weight'])
    best = neighbors[0]
    weight = G[output_node][best]['weight']

    return {
        'mode': 'min_energy',
        'path': [output_node, best],
        'edges': [{'u': output_node, 'v': best, 'weight': weight}],
        'total_energy': weight,
        'energy_budget': None,
    }


def find_longest_path(
    G: nx.Graph,
    output_node: str,
    energy_budget: float,
    max_length: int,
    beam_width: int = 100,
) -> Dict[str, Any]:
    """在 energy_budget 约束下，找节点数最多的路径（beam search 近似）。

    DFS 穷举在 max_length 较大时指数爆炸（51 节点/360 边/深度 10 → 约 10^11 路径）。
    改用 beam search：
        每层保留最多 beam_width 条候选路径（长度优先、能量次之），
        复杂度 O(max_length × beam_width × avg_degree)，与 max_length 线性相关。
    """
    if output_node not in G:
        return _empty_result('longest_path', output_node, energy_budget)

    if energy_budget <= 0:
        print(f"[output_layer] Warning: energy_budget={energy_budget:.4f} <= 0, "
              f"longest_path returns only the output node")
        return _empty_result('longest_path', output_node, energy_budget)

    beam = [([output_node], 0.0, {output_node})]
    best_path = [output_node]
    best_energy = 0.0

    for _ in range(max_length - 1):
        candidates = []
        for path, energy, visited in beam:
            last = path[-1]
            for nb in G.neighbors(last):
                if nb in visited:
                    continue
                w = G[last][nb]['weight']
                if energy + w > energy_budget:
                    continue
                candidates.append((path + [nb], energy + w, visited | {nb}))

        if not candidates:
            break

        candidates.sort(key=lambda x: (-len(x[0]), x[1]))
        beam = candidates[:beam_width]

        for path, energy, _ in beam:
            if (len(path) > len(best_path) or
                    (len(path) == len(best_path) and energy < best_energy)):
                best_path = path
                best_energy = energy

        if len(best_path) >= max_length:
            break

    edges = []
    for i in range(len(best_path) - 1):
        u, v = best_path[i], best_path[i + 1]
        if G.has_edge(u, v):
            edges.append({'u': u, 'v': v, 'weight': G[u][v]['weight']})

    return {
        'mode': 'longest_path',
        'path': best_path,
        'edges': edges,
        'total_energy': best_energy,
        'energy_budget': energy_budget,
    }


def traverse_from_output(
    G: nx.Graph,
    output_node: str,
    traversal_type: str,
    hard_ratio: float,
    steps: int,
    energy_budget: float,
) -> Dict[str, Any]:
    """从 output_node 出发，按软硬遍历走 steps 步，累计边权 ≤ energy_budget。"""
    if output_node not in G:
        return _empty_result('traversal', output_node, energy_budget)

    if energy_budget <= 0:
        print(f"[output_layer] Warning: energy_budget={energy_budget:.4f} <= 0, "
              f"traversal returns only the output node")

    path = [output_node]
    total = 0.0
    current = output_node

    for _ in range(steps):
        neighbors = [n for n in G.neighbors(current) if n not in path]
        if not neighbors:
            break

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

    edges = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if G.has_edge(u, v):
            edges.append({'u': u, 'v': v, 'weight': G[u][v]['weight']})

    return {
        'mode': 'traversal',
        'path': path,
        'edges': edges,
        'total_energy': total,
        'energy_budget': energy_budget,
    }