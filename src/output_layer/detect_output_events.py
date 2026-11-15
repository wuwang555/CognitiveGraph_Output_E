# -*- coding: utf-8 -*-
"""复用原论文 EmergenceDetectorFixed，过滤出"包含输出节点"的压缩/迁移事件。"""

from typing import Any, List

import pandas as pd

from .config_output import OUTPUT_NODE_NAME


_COMPRESSION_COLUMNS = [
    'center', 'related_nodes', 'energy_synergy', 'cohesion',
    'emergence_strength', 'cluster_size', 'avg_connection_strength',
    'compression_potential',
]

_MIGRATION_COLUMNS = [
    'type', 'principle_node', 'from_node', 'to_node',
    'efficiency_gain', 'path', 'emergence_iteration', 'domain_span',
]


def detect_compression_with_output(
    G,
    energy_history: List[float],
    traversal_history: List[Any],
    output_node: str = OUTPUT_NODE_NAME,
    synergy_threshold: float = 0.76,
) -> pd.DataFrame:
    """检测包含输出节点的压缩事件。

    注意：EmergenceDetectorFixed 的 detection_thresholds 是"全量替换"而非合并。
    因此先用默认值构造，再覆盖单个 key。
    """
    from src.emergence.detector_fixed import EmergenceDetectorFixed

    detector = EmergenceDetectorFixed()
    detector.thresholds['compression_synergy'] = synergy_threshold

    all_compressions = detector.detect_spontaneous_compression(
        G, energy_history, traversal_history
    )
    filtered = [
        c for c in all_compressions
        if c.get('center') == output_node
        or output_node in c.get('related_nodes', [])
    ]
    if not filtered:
        return pd.DataFrame(columns=_COMPRESSION_COLUMNS)
    return pd.DataFrame(filtered)


def detect_migration_with_output(
    G,
    traversal_history: List[Any],
    current_iteration: int,
    output_node: str = OUTPUT_NODE_NAME,
    efficiency_threshold: float = 0.35,
) -> pd.DataFrame:
    """检测包含输出节点的迁移事件。"""
    from src.emergence.detector_fixed import EmergenceDetectorFixed

    detector = EmergenceDetectorFixed()
    detector.thresholds['migration_efficiency'] = efficiency_threshold

    all_migrations = detector.detect_emergent_migration(
        G, traversal_history, current_iteration
    )
    filtered = [m for m in all_migrations if output_node in m.get('path', [])]
    if not filtered:
        return pd.DataFrame(columns=_MIGRATION_COLUMNS)
    return pd.DataFrame(filtered)