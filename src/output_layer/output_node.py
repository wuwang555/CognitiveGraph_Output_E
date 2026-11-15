# -*- coding: utf-8 -*-
"""输出节点：相似度计算与边构造。

本模块只负责"计算 context 与已有概念的相似度"和"把相似度转成边"。
不修改原论文文件。
"""

import os
import io
import contextlib
from typing import Dict, List, Tuple

from .config_output import OUTPUT_NODE_NAME


def build_context_similarity(
    context: str,
    existing_concepts: List[str],
    concept_config: str = "B",
) -> Dict[str, float]:
    """计算 context 与每个已有概念的语义相似度。

    实现：
        1. 构造 SemanticConceptNetwork
        2. 把 existing_concepts 的定义注册进去（从 config.CORE_CONCEPT_DEFINITIONS 读）
        3. 把 OUTPUT_NODE_NAME 的定义 = context 注册进去
        4. 对每个 concept 调 calculate_semantic_similarity(OUTPUT_NODE_NAME, concept)
    """
    os.environ['CONFIG_METHOD'] = concept_config.upper()

    from config import CORE_CONCEPT_DEFINITIONS
    from src.core.semantic_network import SemanticConceptNetwork

    net = SemanticConceptNetwork()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for concept in existing_concepts:
            if concept in CORE_CONCEPT_DEFINITIONS:
                net.add_concept_definition(
                    concept, CORE_CONCEPT_DEFINITIONS[concept], "existing"
                )
        net.add_concept_definition(OUTPUT_NODE_NAME, context, "output")

    similarities: Dict[str, float] = {}
    for concept in existing_concepts:
        if concept in net.concept_definitions:
            sim = net.calculate_semantic_similarity(OUTPUT_NODE_NAME, concept)
            similarities[concept] = sim
    return similarities


def build_output_node_edges(
    similarities: Dict[str, float],
    similarity_threshold: float = 0.0,
    skip_zero_similarity: bool = False,
) -> List[Tuple[str, float]]:
    """把相似度转成边列表。

    默认全连接（similarity_threshold=0.0）：输出节点与所有已有概念连边，
    保持原图"稠密图"的基调。这修复了此前因照搬概念-概念 0.1 阈值、
    导致输出节点在 context 与定义词无交集时变成孤立节点的问题。

    边权公式（与原论文一致）：energy = 2.0 - similarity * 1.5，截断到 [0.3, 2.0]。
    当 sim=0 时 energy=2.0（最高能量），符合"无关联 = 高能耗"的物理直觉。

    :param similarities: {concept: similarity}
    :param similarity_threshold: 相似度过滤阈值；0.0 表示全连接
    :param skip_zero_similarity: 是否跳过 sim 严格为 0 的边；默认 False
    :return: [(concept, energy), ...]
    """
    edges: List[Tuple[str, float]] = []
    for concept, sim in similarities.items():
        if sim <= similarity_threshold:
            continue
        if skip_zero_similarity and sim == 0.0:
            continue
        energy = 2.0 - sim * 1.5
        energy = max(0.3, min(2.0, energy))
        edges.append((concept, energy))
    return edges