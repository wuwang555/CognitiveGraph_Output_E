# -*- coding: utf-8 -*-
"""输出层入口脚本。

用法示例：
    # 基线（不插入）
    python -m src.output_layer.run_output --no-insert

    # 插入 + 最小能量（默认全连接）
    python -m src.output_layer.run_output --insert-at 1000 \
        --context-preset algorithms --output-mode min_energy

    # 插入 + 动态预算（mean 度量）
    python -m src.output_layer.run_output --insert-at 1000 \
        --context-preset algorithms --output-mode longest_path \
        --budget-mode dynamic --budget-metric mean --max-path-length 10

    # 插入 + 软硬遍历
    python -m src.output_layer.run_output --insert-at 1000 \
        --context-preset os_concurrency --output-mode traversal \
        --traversal-type mixed --traversal-steps 4

    # 恢复"阈值过滤"行为（非默认）
    python -m src.output_layer.run_output --insert-at 1000 \
        --context-preset algorithms --output-node-threshold 0.1
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Output Layer Runner")

    # 基础
    parser.add_argument('--num-concepts', type=int, default=None)
    parser.add_argument('--concept-config', type=str, default=None,
                        choices=['A', 'B', 'C'])
    parser.add_argument('--base-iterations', type=int, default=None)
    parser.add_argument('--random-seed', type=int, default=None)

    # 插入
    parser.add_argument('--no-insert', action='store_true')
    parser.add_argument('--insert-at', type=int, default=None)
    parser.add_argument('--context', type=str, default=None,
                        help='输出节点上下文定义（优先级高于 --context-preset）')
    parser.add_argument('--context-preset', type=str, default=None,
                        choices=['algorithms', 'databases', 'deep_learning',
                                 'os_concurrency', 'network_distributed',
                                 'compiler_languages'],
                        help='使用预设上下文（针对概念集 B）')
    parser.add_argument('--post-insertion-iterations', type=int, default=None)
    parser.add_argument('--max-iterations', type=int, default=None)

    # 输出节点连边
    parser.add_argument('--output-node-threshold', type=float, default=None,
                        help='相似度阈值；默认 0.0 = 全连接。设 >0 恢复过滤行为。')
    parser.add_argument('--skip-zero-similarity', action='store_true',
                        help='跳过 sim=0 的边（默认全连，包含 sim=0）')

    # 输出提取
    parser.add_argument('--output-mode', type=str, default=None,
                        choices=['min_energy', 'longest_path', 'traversal'])
    parser.add_argument('--energy-budget', type=float, default=None)
    parser.add_argument('--budget-mode', type=str, default=None,
                        choices=['fixed', 'dynamic'])
    parser.add_argument('--budget-metric', type=str, default=None,
                        choices=['total', 'mean'])
    parser.add_argument('--budget-reference-point', type=str, default=None,
                        choices=['pre', 'post'])
    parser.add_argument('--budget-fallback', type=float, default=None)
    parser.add_argument('--max-path-length', type=int, default=None)
    parser.add_argument('--beam-width', type=int, default=None)
    parser.add_argument('--traversal-type', type=str, default=None,
                        choices=['hard', 'soft', 'mixed'])
    parser.add_argument('--hard-ratio', type=float, default=None)
    parser.add_argument('--traversal-steps', type=int, default=None)

    # 检测
    parser.add_argument('--no-detect-compression', action='store_true')
    parser.add_argument('--no-detect-migration', action='store_true')

    # 其他
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    concept_config = (args.concept_config or 'B').upper()
    os.environ['CONFIG_METHOD'] = concept_config

    from src.output_layer.config_output import DEFAULT_OUTPUT_CONFIG, PRESET_CONTEXTS
    from src.output_layer.run_with_output import run_pipeline

    config = DEFAULT_OUTPUT_CONFIG.copy()
    config['concept_config'] = concept_config

    # ---- 基础 ----
    if args.num_concepts is not None:
        config['num_concepts'] = args.num_concepts
    if args.base_iterations is not None:
        config['base_iterations'] = args.base_iterations
    if args.random_seed is not None:
        config['random_seed'] = args.random_seed

    # ---- 插入 ----
    if args.no_insert:
        config['insertion_enabled'] = False
        config['insertion_iteration'] = None
    elif args.insert_at is not None:
        config['insertion_enabled'] = True
        config['insertion_iteration'] = args.insert_at

    # context 优先级：--context > --context-preset > config['context']
    if args.context is not None:
        config['context'] = args.context
    elif args.context_preset is not None:
        config['context'] = PRESET_CONTEXTS[args.context_preset]

    if args.post_insertion_iterations is not None:
        config['post_insertion_iterations'] = args.post_insertion_iterations
    if args.max_iterations is not None:
        config['max_iterations'] = args.max_iterations

    # ---- 输出节点连边 ----
    if args.output_node_threshold is not None:
        config['output_node_similarity_threshold'] = args.output_node_threshold
    if args.skip_zero_similarity:
        config['output_node_skip_zero_similarity'] = True

    # ---- 输出提取 ----
    if args.output_mode is not None:
        config['output_mode'] = args.output_mode
    if args.energy_budget is not None:
        config['energy_budget'] = args.energy_budget
    if args.budget_mode is not None:
        config['budget_mode'] = args.budget_mode
    if args.budget_metric is not None:
        config['budget_metric'] = args.budget_metric
    if args.budget_reference_point is not None:
        config['budget_reference_point'] = args.budget_reference_point
    if args.budget_fallback is not None:
        config['budget_fallback'] = args.budget_fallback
    if args.max_path_length is not None:
        config['max_path_length'] = args.max_path_length
    if args.beam_width is not None:
        config['beam_width'] = args.beam_width
    if args.traversal_type is not None:
        config['traversal_type'] = args.traversal_type
    if args.hard_ratio is not None:
        config['hard_ratio'] = args.hard_ratio
    if args.traversal_steps is not None:
        config['traversal_steps'] = args.traversal_steps

    # ---- 检测 ----
    if args.no_detect_compression:
        config['detect_compression'] = False
    if args.no_detect_migration:
        config['detect_migration'] = False

    result = run_pipeline(config)

    print("\n=== Output Layer Done ===")
    print(f"Scenario dir: {result['scenario_dir']}")
    print(f"Iteration plan: {result['plan']}")
    if result['insertion_info']:
        ii = result['insertion_info']
        print(f"Output node edges: {ii['n_edges_added']} / {ii.get('n_edges_total')} "
              f"(sim range: {ii['similarity_range']['min']:.4f} ~ "
              f"{ii['similarity_range']['max']:.4f}, "
              f"mean {ii['similarity_range']['mean']:.4f})")
    if result['budget_info']:
        bi = result['budget_info']
        print(f"Budget: {bi['budget']:.4f} (mode={bi['mode']}, "
              f"metric={bi.get('metric')}, "
              f"E_insert={bi['E_insert']:.4f}, E_final={bi['E_final']:.4f})")
    if result['output_result']:
        print(f"Output path: {' -> '.join(result['output_result']['path'])}")
        print(f"Output total energy: {result['output_result']['total_energy']:.4f}")
    print(f"Compressions: {result['n_compressions']}")
    print(f"Migrations: {result['n_migrations']}")


if __name__ == "__main__":
    main()