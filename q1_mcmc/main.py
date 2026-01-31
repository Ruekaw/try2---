# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C - Q1: MCMC粉丝投票反推模型
主启动器 (main.py)

一键运行完整推断流程
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import MCMCConfig, PathConfig, FilterConfig
from engine import create_engine
from diagnostics import generate_diagnostic_report, HAS_MATPLOTLIB

if HAS_MATPLOTLIB:
    from diagnostics import plot_ppc_summary, plot_certainty_heatmap
    import matplotlib.pyplot as plt


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="DWTS 粉丝投票 MCMC 反推模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据路径
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="outputs/dwts_long_clean.csv",
        help="输入数据文件路径（相对于工作目录）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/q1_mcmc",
        help="输出目录路径（相对于工作目录）"
    )
    
    # MCMC 参数
    parser.add_argument(
        "--n-samples", "-n",
        type=int,
        default=5000,
        help="保留的有效样本数"
    )
    parser.add_argument(
        "--burn-in", "-b",
        type=int,
        default=2000,
        help="预热期样本数"
    )
    parser.add_argument(
        "--thin", "-t",
        type=int,
        default=2,
        help="稀疏采样间隔"
    )
    parser.add_argument(
        "--lambda", dest="violation_lambda",
        type=float,
        default=None,
        help="统一的软约束惩罚强度（设置后将覆盖分赛制 lambda）"
    )
    parser.add_argument(
        "--lambda-percent",
        type=float,
        default=50.0,
        help="百分比制（S3-27）的软约束惩罚强度"
    )
    parser.add_argument(
        "--lambda-rank",
        type=float,
        default=3.0,
        help="排名制（S1-2, S28+）的软约束惩罚强度"
    )
    parser.add_argument(
        "--no-method-specific-lambda",
        action="store_true",
        help="禁用按赛制自适应 lambda，改用统一 --lambda"
    )
    parser.add_argument(
        "--proposal-scale",
        type=float,
        default=100.0,
        help="提议分布浓度缩放"
    )
    parser.add_argument(
        "--prior-alpha",
        type=float,
        default=1.0,
        help="Dirichlet先验浓度"
    )
    
    # 约束模式
    parser.add_argument(
        "--hard-constraint",
        action="store_true",
        help="使用硬约束模式（默认软约束）"
    )
    parser.add_argument(
        "--no-judge-save",
        action="store_true",
        help="禁用评委救人机制"
    )
    
    # 过滤参数
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="赛季范围，格式: start-end，例如 1-34"
    )
    parser.add_argument(
        "--exclude-seasons",
        type=str,
        default="",
        help="排除的赛季，用逗号分隔，例如 15（默认不排除）"
    )
    
    # 并行参数
    parser.add_argument(
        "--n-jobs", "-j",
        type=int,
        default=None,
        help="并行进程数，默认为 CPU核心数-1，设为1则串行"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="禁用并行，使用串行模式"
    )
    
    # 其他
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="不生成可视化图表"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("MCM 2026 Problem C - Q1")
    print("MCMC 粉丝投票反推模型")
    print("=" * 60)
    print()
    
    start_time = time.time()
    
    # === 配置 ===
    
    # === 软约束 λ：按赛制自适应（默认）===
    use_method_specific_lambda = not args.no_method_specific_lambda
    if args.violation_lambda is not None:
        # 显式指定统一 lambda 时：覆盖分赛制 lambda
        use_method_specific_lambda = False
        lambda_percent = float(args.violation_lambda)
        lambda_rank = float(args.violation_lambda)
        lambda_global = float(args.violation_lambda)
    else:
        lambda_percent = float(args.lambda_percent)
        lambda_rank = float(args.lambda_rank)
        # 历史字段保留：当关闭按赛制自适应时使用
        lambda_global = float(args.lambda_percent)

    # MCMC 配置
    mcmc_config = MCMCConfig(
        n_samples=args.n_samples,
        burn_in=args.burn_in,
        thin=args.thin,
        proposal_scale=args.proposal_scale,
        prior_alpha=args.prior_alpha,
        soft_elimination=not args.hard_constraint,
        violation_lambda=lambda_global,
        use_method_specific_lambda=use_method_specific_lambda,
        violation_lambda_percent=lambda_percent,
        violation_lambda_rank=lambda_rank,
        judge_save_enabled=not args.no_judge_save,
        random_seed=args.seed
    )
    
    # 路径配置
    workspace = Path(__file__).parent.parent
    path_config = PathConfig(
        workspace=workspace,
        input_csv=args.input,
        output_dir=args.output
    )
    
    # 过滤配置
    season_range = None
    if args.seasons:
        parts = args.seasons.split("-")
        season_range = (int(parts[0]), int(parts[1]))
    
    exclude_seasons = []
    if args.exclude_seasons:
        exclude_seasons = [int(s.strip()) for s in args.exclude_seasons.split(",") if s.strip()]
    
    filter_config = FilterConfig(
        season_range=season_range,
        exclude_seasons=exclude_seasons
    )
    
    # 并行配置
    import os
    n_jobs = args.n_jobs
    use_parallel = not args.no_parallel
    
    if n_jobs is None:
        cpu = os.cpu_count() or 1
        n_jobs = max(1, cpu - 1)
    
    # === 打印配置 ===
    
    print("配置信息:")
    print(f"  输入文件: {path_config.get_input_path()}")
    print(f"  输出目录: {path_config.get_output_dir()}")
    print(f"  样本数: {mcmc_config.n_samples}")
    print(f"  预热期: {mcmc_config.burn_in}")
    print(f"  稀疏间隔: {mcmc_config.thin}")
    if mcmc_config.use_method_specific_lambda:
        print("  惩罚强度: 按赛制自适应")
        print(f"    percent λ: {mcmc_config.violation_lambda_percent}")
        print(f"    rank    λ: {mcmc_config.violation_lambda_rank}")
    else:
        print(f"  惩罚强度(统一λ): {mcmc_config.violation_lambda}")
    print(f"  约束模式: {'硬约束' if args.hard_constraint else '软约束'}")
    print(f"  评委救人: {'启用' if not args.no_judge_save else '禁用'}")
    if season_range:
        print(f"  赛季范围: {season_range[0]}-{season_range[1]}")
    if exclude_seasons:
        print(f"  排除赛季: {exclude_seasons}")
    print(f"  随机种子: {mcmc_config.random_seed}")
    print(f"  并行模式: {'启用' if use_parallel else '禁用'} ({n_jobs} 进程)")
    print()
    
    # === 创建引擎并运行 ===
    
    engine = create_engine(mcmc_config, path_config, filter_config)
    
    print("加载数据...")
    engine.load_data()
    print(f"  已加载 {len(engine.data)} 行数据")
    print(f"  赛季: {sorted(engine.data['season'].unique())}")
    print()
    
    print("开始 MCMC 推断...")
    engine.infer_all(n_jobs=n_jobs, use_parallel=use_parallel)
    
    inference_time = time.time() - start_time
    print(f"\n推断完成，耗时: {inference_time:.1f} 秒")
    print()
    
    # === 导出结果 ===
    
    print("导出结果...")
    long_df, wide_df, summary = engine.export_results()
    
    # === 生成诊断报告 ===
    
    print("\n生成诊断报告...")
    report = generate_diagnostic_report(long_df, str(path_config.get_output_dir()))
    
    if args.verbose:
        print("\n" + report)
    
    # === 可视化 ===
    
    if HAS_MATPLOTLIB and not args.no_plots:
        print("\n生成可视化图表...")
        output_dir = path_config.get_output_dir()
        
        # PPC 汇总图
        fig = plot_ppc_summary(long_df, title="PPC Consistency by Season")
        if fig:
            fig.savefig(output_dir / "ppc_summary.png", dpi=150, bbox_inches='tight')
            print(f"  已保存: ppc_summary.png")
            plt.close(fig)
        
        # 为每个赛季生成确定性热力图（可选，这里只生成前几个）
        seasons_to_plot = sorted(engine.results.keys())[:3]  # 只画前3个赛季
        for season in seasons_to_plot:
            season_df = long_df[long_df['season'] == season]
            if len(season_df) > 0:
                fig = plot_certainty_heatmap(season_df, season)
                if fig:
                    fig.savefig(output_dir / f"certainty_heatmap_s{season}.png", dpi=150, bbox_inches='tight')
                    print(f"  已保存: certainty_heatmap_s{season}.png")
                    plt.close(fig)
    
    # === 完成 ===
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("运行完成!")
    print("=" * 60)
    print(f"总耗时: {total_time:.1f} 秒")
    print(f"推断赛季数: {len(engine.results)}")
    print(f"推断周数: {sum(sr.n_weeks for sr in engine.results.values())}")
    print(f"平均 PPC 一致性: {summary['mean_ppc_consistency']:.3f}")
    print(f"平均接受率: {summary['mean_acceptance_rate']:.3f}")
    print()
    print("输出文件:")
    print(f"  - {path_config.get_output_dir() / path_config.output_long}")
    print(f"  - {path_config.get_output_dir() / path_config.output_wide}")
    print(f"  - {path_config.get_output_dir() / path_config.output_summary}")
    print(f"  - {path_config.get_output_dir() / 'diagnostic_report.txt'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
