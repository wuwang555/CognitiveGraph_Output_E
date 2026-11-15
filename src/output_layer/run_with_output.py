# -*- coding: utf-8 -*-
"""主流程：跑 G1 → 插入 → 继续跑 → 计算动态预算 → 提取 → 检测 → 落盘。"""

import contextlib
import csv
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .config_output import OUTPUT_NODE_NAME
from .detect_output_events import (
    detect_compression_with_output,
    detect_migration_with_output,
)
from .extract_output import extract_output
from .insert_node import compute_iteration_plan, insert_output_node_into_universe


_SUMMARY_HEADER = (
    'timestamp,concept_config,num_concepts,insertion_enabled,'
    'insertion_iteration,total_iterations,output_mode,energy_budget,'
    'n_output_edges,output_path,output_total_energy,n_compressions,n_migrations'
)


def _build_initial_network(universe, num_concepts):
    """用 num_concepts 构建初始网络（等价于 initialize_semantic_network 但支持 num_concepts）。"""
    from src.core.semantic_network import SemanticConceptNetwork

    semantic_net = SemanticConceptNetwork()
    semantic_net.build_comprehensive_network(num_concepts=num_concepts)

    nodes = list(semantic_net.concept_definitions.keys())
    universe.G.add_nodes_from(nodes)

    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i + 1:], i + 1):
            similarity = semantic_net.calculate_semantic_similarity(node1, node2)
            if similarity > 0.1:
                energy = max(0.3, min(2.0, 2.0 - similarity * 1.5))
                universe.G.add_edge(
                    node1, node2,
                    weight=energy,
                    original_weight=energy,
                    similarity=similarity,
                    traversal_count=0,
                )
                universe.last_activation_time[(node1, node2)] = 0


def _network_energy(G, metric: str = 'total') -> float:
    """按 metric 计算网络能量。

    metric='total': Σ weight
    metric='mean' : mean(weight)
    """
    if G.number_of_edges() == 0:
        return 0.0
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    if metric == 'total':
        return float(sum(weights))
    elif metric == 'mean':
        return float(sum(weights) / len(weights))
    else:
        raise ValueError(f"Unknown budget_metric: {metric}")


def _compute_dynamic_budget(universe, E_insert: float, config: dict) -> dict:
    """计算动态能量预算：budget = E_insert - E_final。"""
    metric = config.get('budget_metric', 'total')
    E_final = _network_energy(universe.G, metric)

    raw_diff = E_insert - E_final
    fallback_used = False
    if raw_diff <= 0:
        fb = config.get('budget_fallback', 5.0)
        print(f"[output_layer] Warning: dynamic budget <= 0 "
              f"(E_insert={E_insert:.4f}, E_final={E_final:.4f}). "
              f"Using fallback={fb}")
        budget = fb
        fallback_used = True
    else:
        budget = raw_diff

    info = {
        'mode': 'dynamic',
        'metric': metric,
        'reference_point': config.get('budget_reference_point', 'post'),
        'E_insert': E_insert,
        'E_final': E_final,
        'raw_diff': raw_diff,
        'budget': budget,
        'fallback_used': fallback_used,
        'n_edges_insert': universe.G.number_of_edges(),
        'n_edges_final': universe.G.number_of_edges(),
    }
    return {'budget': budget, 'info': info}


def _patch_bidirectional_activation(universe, output_node: str) -> None:
    """实例级补丁：输出节点的边被激活时，同时写入两个方向的 key。"""
    original_record = universe.record_edge_activation

    def record_bidirectional(u, v):
        original_record(u, v)
        if u == output_node or v == output_node:
            t = universe.last_activation_time.get((u, v), 0)
            universe.last_activation_time[(v, u)] = t

    universe.record_edge_activation = record_bidirectional


def _prepare_summary_file(output_root: Path, timestamp: str) -> Path:
    """准备 summary.csv：列格式变化时备份旧文件。"""
    summary_path = output_root / 'summary.csv'
    if summary_path.exists():
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                header = f.readline().strip()
        except Exception:
            header = ''
        if header != _SUMMARY_HEADER:
            legacy_path = output_root / f"summary_legacy_{timestamp}.csv"
            summary_path.rename(legacy_path)
            print(f"[output_layer] Old summary.csv backed up to {legacy_path.name}")
    return summary_path


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """完整流程。"""
    concept_config = config.get('concept_config', 'B').upper()
    os.environ['CONFIG_METHOD'] = concept_config

    from src.emergence.universe import CognitiveUniverse

    with contextlib.redirect_stdout(io.StringIO()):
        universe = CognitiveUniverse(network_seed=config.get('random_seed', 42))
        _build_initial_network(universe, config.get('num_concepts', 51))

    plan = compute_iteration_plan(
        base_iterations=config.get('base_iterations', 10000),
        insertion_enabled=config.get('insertion_enabled', True),
        insertion_iteration=config.get('insertion_iteration', None),
        post_insertion_iterations=config.get('post_insertion_iterations', 10000),
        max_iterations=config.get('max_iterations', None),
    )

    if plan['pre_insertion'] > 0:
        with contextlib.redirect_stdout(io.StringIO()):
            universe.evolve(iterations=plan['pre_insertion'], observation_interval=100)

    insertion_info = None
    E_insert = None
    budget_info = None
    metric = config.get('budget_metric', 'total')
    ref_point = config.get('budget_reference_point', 'post')

    if plan['will_insert']:
        context = config.get('context')
        if context is None:
            print(
                "\n[output_layer] ⚠️  WARNING: config['context'] is None.\n"
                "  将使用 fallback 占位文本 '我想说一句关于电脑的话'。\n"
                "  由于现在默认全连接，输出节点仍会连边，但边权会比较高（sim=0 → energy=2.0）。\n"
                "  建议通过 --context-preset <名> 或 --context '<文本>' 提供有效 context。\n"
            )
            context = "我想说一句关于电脑的话"

        E_pre = _network_energy(universe.G, metric) if ref_point == 'pre' else None

        insertion_info = insert_output_node_into_universe(
            universe, context, concept_config,
            similarity_threshold=config.get('output_node_similarity_threshold', 0.0),
            skip_zero_similarity=config.get('output_node_skip_zero_similarity', False),
        )

        _patch_bidirectional_activation(universe, OUTPUT_NODE_NAME)

        if ref_point == 'post':
            E_insert = _network_energy(universe.G, metric)
        else:
            E_insert = E_pre

    extract_config = dict(config)
    if plan['will_insert'] and plan['post_insertion'] > 0:
        with contextlib.redirect_stdout(io.StringIO()):
            universe.evolve(iterations=plan['post_insertion'], observation_interval=100)

    if plan['will_insert']:
        if config.get('budget_mode', 'fixed') == 'dynamic':
            result = _compute_dynamic_budget(universe, E_insert, config)
            budget_info = result['info']
            extract_config['energy_budget'] = result['budget']
        else:
            budget_info = {
                'mode': 'fixed',
                'metric': None,
                'reference_point': None,
                'E_insert': E_insert,
                'E_final': _network_energy(universe.G, metric),
                'raw_diff': None,
                'budget': config.get('energy_budget', 5.0),
                'fallback_used': False,
                'n_edges_insert': universe.G.number_of_edges(),
                'n_edges_final': universe.G.number_of_edges(),
            }
            extract_config['energy_budget'] = config.get('energy_budget', 5.0)

    output_result = None
    if plan['will_insert']:
        output_result = extract_output(universe.G, OUTPUT_NODE_NAME, extract_config)

    compressions_df = None
    migrations_df = None
    if plan['will_insert']:
        if config.get('detect_compression', True):
            compressions_df = detect_compression_with_output(
                universe.G, universe.energy_history, universe.traversal_history,
                OUTPUT_NODE_NAME, config.get('compression_synergy', 0.76),
            )
        if config.get('detect_migration', True):
            migrations_df = detect_migration_with_output(
                universe.G, universe.traversal_history, universe.iteration_count,
                OUTPUT_NODE_NAME, config.get('migration_efficiency', 0.35),
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(config.get('output_root', 'results/output_layer'))
    scenario_dir = output_root / f"scenario_{timestamp}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    plan_out = dict(plan)
    if budget_info is not None:
        plan_out['energy_budget'] = budget_info['budget']
        plan_out['budget_metric'] = budget_info.get('metric', 'fixed')
        plan_out['budget_reference_point'] = budget_info.get('reference_point')
    with open(scenario_dir / 'iteration_plan.json', 'w', encoding='utf-8') as f:
        json.dump(plan_out, f, ensure_ascii=False, indent=2)

    if budget_info is not None:
        with open(scenario_dir / 'energy_budget.json', 'w', encoding='utf-8') as f:
            json.dump(budget_info, f, ensure_ascii=False, indent=2)

    if insertion_info is not None:
        with open(scenario_dir / 'output_node_info.json', 'w', encoding='utf-8') as f:
            json.dump({
                'output_node': insertion_info['output_node'],
                'context': insertion_info['context'],
                'n_edges_added': insertion_info['n_edges_added'],
                'n_edges_total': insertion_info.get('n_edges_total'),
                'similarity_range': insertion_info.get('similarity_range'),
            }, f, ensure_ascii=False, indent=2)

        with open(scenario_dir / 'output_edges.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['concept', 'energy', 'similarity'])
            for concept, energy in insertion_info['edges']:
                sim = universe.G[OUTPUT_NODE_NAME][concept].get('similarity', 0.0)
                writer.writerow([concept, energy, sim])

    if output_result is not None:
        with open(scenario_dir / 'output_result.json', 'w', encoding='utf-8') as f:
            json.dump(output_result, f, ensure_ascii=False, indent=2)

    if compressions_df is not None:
        compressions_df.to_csv(scenario_dir / 'output_compressions.csv',
                               index=False, encoding='utf-8')
    if migrations_df is not None:
        migrations_df.to_csv(scenario_dir / 'output_migrations.csv',
                             index=False, encoding='utf-8')

    weights = [universe.G[u][v]['weight'] for u, v in universe.G.edges()]
    mean_weight = float(sum(weights) / len(weights)) if weights else 0.0
    with open(scenario_dir / 'g1_summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'n_nodes': universe.G.number_of_nodes(),
            'n_edges': universe.G.number_of_edges(),
            'mean_weight': mean_weight,
            'total_weight': float(sum(weights)) if weights else 0.0,
            'final_energy': universe.calculate_network_energy(),
            'final_iteration': universe.iteration_count,
        }, f, ensure_ascii=False, indent=2)

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = _prepare_summary_file(output_root, timestamp)
    summary_exists = summary_path.exists()
    with open(summary_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow(_SUMMARY_HEADER.split(','))
        writer.writerow([
            timestamp,
            concept_config,
            config.get('num_concepts', 51),
            config.get('insertion_enabled', True),
            config.get('insertion_iteration') if config.get('insertion_enabled') else None,
            plan['total'],
            config.get('output_mode', 'min_energy'),
            budget_info['budget'] if budget_info is not None else 0.0,
            insertion_info['n_edges_added'] if insertion_info else 0,
            ' -> '.join(output_result['path']) if output_result else '',
            output_result['total_energy'] if output_result else 0.0,
            len(compressions_df) if compressions_df is not None else 0,
            len(migrations_df) if migrations_df is not None else 0,
        ])

    return {
        'scenario_dir': str(scenario_dir),
        'plan': plan,
        'insertion_info': insertion_info,
        'budget_info': budget_info,
        'output_result': output_result,
        'n_compressions': len(compressions_df) if compressions_df is not None else 0,
        'n_migrations': len(migrations_df) if migrations_df is not None else 0,
    }