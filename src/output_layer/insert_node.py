# -*- coding: utf-8 -*-
"""在 G1 中插入输出节点，并计算迭代计划。"""

from typing import Any, Dict, Optional

from .config_output import OUTPUT_NODE_NAME
from .output_node import build_context_similarity, build_output_node_edges


def compute_iteration_plan(
    base_iterations: int,
    insertion_enabled: bool,
    insertion_iteration: Optional[int],
    post_insertion_iterations: int,
    max_iterations: Optional[int],
) -> Dict[str, Any]:
    """计算迭代计划（规则见 agent.md §1.2）。"""
    if not insertion_enabled or insertion_iteration is None:
        return {
            'pre_insertion': base_iterations,
            'post_insertion': 0,
            'total': base_iterations,
            'will_insert': False,
        }

    T = insertion_iteration
    if T < 0:
        print(f"[output_layer] Warning: insertion_iteration={T} < 0, clamped to 0")
        T = 0
    if T > base_iterations:
        print(f"[output_layer] Warning: insertion_iteration={T} > base_iterations={base_iterations}, clamped")
        T = base_iterations

    total = T + post_insertion_iterations
    if max_iterations is not None:
        total = min(total, max_iterations)

    if T > total:
        print(f"[output_layer] Warning: max_iterations < insertion_iteration, "
              f"insertion clamped to {total} (post_insertion=0)")
        T = total
    post_actual = total - T

    return {
        'pre_insertion': T,
        'post_insertion': post_actual,
        'total': total,
        'will_insert': True,
    }


def insert_output_node_into_universe(
    universe: Any,
    context: str,
    concept_config: str = "B",
    similarity_threshold: float = 0.0,
    skip_zero_similarity: bool = False,
) -> Dict[str, Any]:
    """在 universe.G 上插入输出节点，并连无向边。

    默认全连接（similarity_threshold=0.0），与所有已有概念连边，
    保持原论文 G1 的稠密图基调。

    边插入方向使用 (concept, OUTPUT_NODE_NAME)，并为两个方向都设置
    last_activation_time，避免遗忘机制的方向不一致问题。
    """
    existing = list(universe.G.nodes())

    sims = build_context_similarity(context, existing, concept_config)
    edges = build_output_node_edges(
        sims, similarity_threshold, skip_zero_similarity
    )

    if len(edges) == 0:
        raise RuntimeError(
            f"\n[output_layer] Output node has 0 edges — 无法插入。\n"
            f"  context (前 120 字): {context[:120]!r}\n"
            f"  similarity_threshold={similarity_threshold}, "
            f"skip_zero_similarity={skip_zero_similarity}\n"
            f"  可能原因：\n"
            f"    1) output_node_similarity_threshold > 0 且所有概念 sim 都低于该阈值\n"
            f"    2) existing_concepts 为空（universe.G 没有节点）\n"
            f"  建议：把 output_node_similarity_threshold 设为 0.0（全连接）重跑。\n"
        )

    universe.G.add_node(OUTPUT_NODE_NAME, is_output=True, original_weight=2.0)

    current_time = universe.iteration_count

    for concept, energy in edges:
        universe.G.add_edge(
            concept, OUTPUT_NODE_NAME,
            weight=energy,
            original_weight=energy,
            similarity=sims[concept],
            traversal_count=0,
        )
        universe.last_activation_time[(concept, OUTPUT_NODE_NAME)] = current_time
        universe.last_activation_time[(OUTPUT_NODE_NAME, concept)] = current_time

    sim_values = list(sims.values()) if sims else [0.0]
    return {
        'output_node': OUTPUT_NODE_NAME,
        'context': context,
        'n_edges_added': len(edges),
        'n_edges_total': len(existing),
        'edges': edges,
        'similarity_range': {
            'min': float(min(sim_values)),
            'max': float(max(sim_values)),
            'mean': float(sum(sim_values) / len(sim_values)),
            'n_above_threshold': len(edges),
        },
    }