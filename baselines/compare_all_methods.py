#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多方法对比可视化脚本

此脚本用于:
1. 读取所有已评估方法的结果 (包括 compared_alg 中的方法和新的 PARSAC/SupeRANSAC)
2. 生成多联对比图 (与论文中的图表格式一致)
3. 生成数据表格并保存到 txt 文件

使用方法:
    python compare_all_methods.py --methods ours ransac kmeans gmm parsac superansac
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))

# ============================================================================
# 配置区域 - 在此编辑要对比的方法列表
# ============================================================================

# 可用的方法列表 (可根据需要注释/取消注释)
AVAILABLE_METHODS = [
    # 优化方法 (自己的方法)
    'ours',
    
    # 传统聚类方法 (from compared_alg/others)
    'ransac',
    'kmeans',
    'gmm',
    'dbscan',
    'optics',
    'agglomerative',
    
    # 优化方法 (from compared_alg/optimization)
    'gurobi',
    'casadi',
    'cvx',
    
    # 新增的基线方法
    'parsac',
    'superansac',
]

# 方法显示名称映射
METHOD_DISPLAY_NAMES = {
    'ours': 'Ours (Manifold Opt.)',
    'ransac': 'RANSAC',
    'kmeans': 'K-Means',
    'gmm': 'GMM',
    'dbscan': 'DBSCAN',
    'optics': 'OPTICS',
    'agglomerative': 'Agglomerative',
    'gurobi': 'Gurobi',
    'casadi': 'CasADi',
    'cvx': 'CVX',
    'parsac': 'PARSAC',
    'superansac': 'SupeRANSAC',
}

# 方法颜色映射
METHOD_COLORS = {
    'ours': '#e41a1c',
    'ransac': '#377eb8',
    'kmeans': '#4daf4a',
    'gmm': '#984ea3',
    'dbscan': '#ff7f00',
    'optics': '#ffff33',
    'agglomerative': '#a65628',
    'gurobi': '#f781bf',
    'casadi': '#999999',
    'cvx': '#66c2a5',
    'parsac': '#8dd3c7',
    'superansac': '#fb8072',
}

# ============================================================================
# 数据加载函数
# ============================================================================

def load_method_results(method: str, results_dir: str, num_samples: int = 20) -> dict:
    """
    加载指定方法的评估结果
    
    Args:
        method: 方法名称
        results_dir: 结果目录
        num_samples: 样本数量
        
    Returns:
        dict: 包含评估指标的字典，如果加载失败返回 None
    """
    # 尝试从 CSV 文件加载
    csv_path = os.path.join(results_dir, method, f"{method}_results.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return {
                'method': method,
                'total_cost': df['total_cost'].values,
                'cost_ratio': df['cost_ratio'].values if 'cost_ratio' in df.columns else None,
                'average_distance': df['average_distance'].values if 'average_distance' in df.columns else None,
                'total_hbar_distance': df['total_hbar_distance'].values,
                'model_count': df['model_count'].values if 'model_count' in df.columns else None,
                'model_count_error': df['model_count_error'].values if 'model_count_error' in df.columns else None,
                'runtime': df['runtime'].values if 'runtime' in df.columns else None,
            }
        except Exception as e:
            print(f"警告: 无法加载 {method} 的结果: {e}")
            return None
    
    return None


def run_method_evaluation(method: str, project_root: str, known_count: bool = True) -> dict:
    """
    运行指定方法的评估（如果结果文件不存在）
    
    Args:
        method: 方法名称
        project_root: 项目根目录
        known_count: 是否已知模型数量
        
    Returns:
        dict: 评估结果
    """
    from data.read_data import read_data_2D
    from evaluate_utils import Hyperplane, Polyhedron, full_evaluate
    
    data_dir = os.path.join(project_root, 'csv_dataset')
    gt_dir = os.path.join(project_root, 'csv_groundtruth')
    
    results = {
        'method': method,
        'total_cost': [],
        'cost_ratio': [],
        'average_distance': [],
        'total_hbar_distance': [],
        'model_count': [],
        'model_count_error': [],
        'runtime': [],
    }
    
    for i in range(20):
        data, gt_data, gt_total_cost = read_data_2D(data_dir, gt_dir, file_index=i)
        
        # 构建真值
        gt_hyperplanes = []
        for j in range(len(gt_data)):
            gt_hyperplanes.append(Hyperplane(normal=gt_data[j, :2], distance=gt_data[j, 2]))
        ground_truth = Polyhedron(dim=2, hyperplanes=gt_hyperplanes)
        
        num_lines = len(gt_hyperplanes) if known_count else None
        
        if method == 'parsac':
            from scripts_for_eval.parsac.line_fitter import SimplePARSACLineFitter
            import time
            fitter = SimplePARSACLineFitter(num_hypotheses=500, inlier_threshold=0.15)
            start = time.time()
            lines = fitter.fit(data, num_lines=num_lines)
            runtime = time.time() - start
            hyperplanes = [Hyperplane(normal=n, distance=d) for n, d in lines]
            
        elif method == 'superansac':
            from scripts_for_eval.superansac.sequential_ransac import SequentialRANSAC2DLine, RANSACConfig
            import time
            config = RANSACConfig(max_iterations=1000, inlier_threshold=0.3, min_inliers=10)
            fitter = SequentialRANSAC2DLine(config=config)
            start = time.time()
            lines = fitter.fit(data, max_lines=num_lines)
            runtime = time.time() - start
            hyperplanes = [Hyperplane(normal=l.normal, distance=l.distance) for l in lines]
            
        elif method == 'ransac':
            from compared_alg.others.RANSAC import ransac
            import time
            alg = ransac(num_lines or 4, 2)
            alg.set_data(data)
            start = time.time()
            alg.solve()
            runtime = time.time() - start
            hyperplanes = [Hyperplane(normal=alg.vectors[k], distance=alg.distances[k]) 
                          for k in range(alg.n)]
            
        elif method == 'kmeans':
            from compared_alg.others.K_Means import Kmeans
            import time
            alg = Kmeans(num_lines or 4, 2)
            alg.set_data(data)
            start = time.time()
            alg.solve()
            runtime = time.time() - start
            hyperplanes = [Hyperplane(normal=alg.vectors[k], distance=alg.distances[k]) 
                          for k in range(alg.n)]
            
        elif method == 'gmm':
            from compared_alg.others.GMM import GMM
            import time
            alg = GMM(num_lines or 4, 2)
            alg.set_data(data)
            start = time.time()
            alg.solve()
            runtime = time.time() - start
            hyperplanes = [Hyperplane(normal=alg.vectors[k], distance=alg.distances[k]) 
                          for k in range(alg.n)]
        else:
            print(f"警告: 方法 {method} 暂不支持自动评估")
            return None
        
        result = Polyhedron(dim=2, hyperplanes=hyperplanes)
        eval_result = full_evaluate(data, ground_truth, gt_total_cost, result, runtime)
        
        results['total_cost'].append(eval_result.total_cost)
        results['cost_ratio'].append(eval_result.cost_ratio)
        results['average_distance'].append(eval_result.average_distance)
        results['total_hbar_distance'].append(eval_result.total_hbar_distance)
        results['model_count'].append(eval_result.model_count)
        results['model_count_error'].append(eval_result.model_count_error)
        results['runtime'].append(eval_result.runtime)
    
    # 转换为 numpy 数组
    for key in results:
        if key != 'method' and results[key]:
            results[key] = np.array(results[key])
    
    return results


# ============================================================================
# 可视化函数
# ============================================================================

def create_comparison_bar_chart(all_results: list, 
                                 metric: str,
                                 title: str,
                                 ylabel: str,
                                 output_path: str,
                                 figsize: tuple = (12, 6)):
    """
    创建柱状对比图
    
    Args:
        all_results: 所有方法的结果列表
        metric: 要绘制的指标名称
        title: 图表标题
        ylabel: Y轴标签
        output_path: 输出路径
        figsize: 图形大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    methods = []
    means = []
    stds = []
    colors = []
    
    for result in all_results:
        if result is None or result[metric] is None:
            continue
        method = result['method']
        methods.append(METHOD_DISPLAY_NAMES.get(method, method))
        means.append(np.mean(result[metric]))
        stds.append(np.std(result[metric]))
        colors.append(METHOD_COLORS.get(method, '#333333'))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    
    # 添加数值标签
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")


def create_multi_panel_figure(all_results: list, output_path: str):
    """
    创建多联图 (与论文中的图表格式一致)
    
    Args:
        all_results: 所有方法的结果列表
        output_path: 输出路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics = [
        ('total_cost', 'Total Cost', 'Sum of Min Distances'),
        ('cost_ratio', 'Cost Ratio', 'Cost / GT Cost'),
        ('total_hbar_distance', 'H-bar Distance', 'Sum of H-bar Distances'),
        ('model_count_error', 'Model Count Error', '|Fitted - GT|'),
        ('runtime', 'Runtime', 'Seconds'),
        ('average_distance', 'Average Distance', 'Cost / N'),
    ]
    
    for idx, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        methods = []
        means = []
        stds = []
        colors = []
        
        for result in all_results:
            if result is None or result.get(metric) is None:
                continue
            method = result['method']
            data = result[metric]
            if data is None or len(data) == 0:
                continue
            methods.append(METHOD_DISPLAY_NAMES.get(method, method))
            means.append(np.mean(data))
            stds.append(np.std(data))
            colors.append(METHOD_COLORS.get(method, '#333333'))
        
        if not methods:
            continue
            
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Multi-Method Comparison on 2D Line Fitting Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存多联图: {output_path}")


def create_box_plot_comparison(all_results: list, output_path: str):
    """
    创建箱线图对比
    
    Args:
        all_results: 所有方法的结果列表
        output_path: 输出路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [
        ('total_cost', 'Total Cost'),
        ('total_hbar_distance', 'H-bar Distance'),
        ('cost_ratio', 'Cost Ratio'),
        ('runtime', 'Runtime (s)'),
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        data_to_plot = []
        labels = []
        
        for result in all_results:
            if result is None or result.get(metric) is None:
                continue
            method = result['method']
            data = result[metric]
            if data is None or len(data) == 0:
                continue
            data_to_plot.append(data)
            labels.append(METHOD_DISPLAY_NAMES.get(method, method))
        
        if not data_to_plot:
            continue
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # 设置颜色
        for i, (box, result) in enumerate(zip(bp['boxes'], all_results)):
            if result is not None:
                color = METHOD_COLORS.get(result['method'], '#333333')
                box.set_facecolor(color)
                box.set_alpha(0.7)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Box Plot Comparison of All Methods', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存箱线图: {output_path}")


# ============================================================================
# 表格生成函数
# ============================================================================

def generate_comparison_table(all_results: list, output_path: str):
    """
    生成对比表格并保存到 txt 文件
    
    Args:
        all_results: 所有方法的结果列表
        output_path: 输出路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("Multi-Method Comparison Results\n")
        f.write("=" * 120 + "\n\n")
        
        # 表头
        headers = ['Method', 'Model Count', 'Total Cost', 'Cost Ratio', 'H-bar Distance', 'Avg Distance', 'Runtime (s)']
        header_line = " | ".join([f"{h:^15}" for h in headers])
        f.write(header_line + "\n")
        f.write("-" * 120 + "\n")
        
        # 数据行
        for result in all_results:
            if result is None:
                continue
            
            method = METHOD_DISPLAY_NAMES.get(result['method'], result['method'])
            
            def format_metric(data, fmt=".4f"):
                if data is None or len(data) == 0:
                    return "N/A"
                return f"{np.mean(data):{fmt}} ± {np.std(data):{fmt}}"
            
            row = [
                f"{method:^15}",
                format_metric(result.get('model_count'), ".2f"),
                format_metric(result.get('total_cost')),
                format_metric(result.get('cost_ratio')),
                format_metric(result.get('total_hbar_distance')),
                format_metric(result.get('average_distance'), ".6f"),
                format_metric(result.get('runtime')),
            ]
            f.write(" | ".join([f"{r:^15}" for r in row]) + "\n")
        
        f.write("-" * 120 + "\n\n")
        
        # LaTeX 格式表格
        f.write("\n" + "=" * 80 + "\n")
        f.write("LaTeX Table Format:\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison of Different Methods on 2D Line Fitting Dataset}\n")
        f.write("\\label{tab:comparison}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Model Count & Total Cost & Cost Ratio & H-bar Dist. & Avg Dist. & Runtime (s) \\\\\n")
        f.write("\\midrule\n")
        
        for result in all_results:
            if result is None:
                continue
            
            method = METHOD_DISPLAY_NAMES.get(result['method'], result['method'])
            
            def format_latex(data, fmt=".3f"):
                if data is None or len(data) == 0:
                    return "-"
                return f"${np.mean(data):{fmt}}$"
            
            row = [
                method,
                format_latex(result.get('model_count'), ".1f"),
                format_latex(result.get('total_cost')),
                format_latex(result.get('cost_ratio')),
                format_latex(result.get('total_hbar_distance')),
                format_latex(result.get('average_distance'), ".5f"),
                format_latex(result.get('runtime')),
            ]
            f.write(" & ".join(row) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"已保存对比表格: {output_path}")


def generate_detailed_table(all_results: list, output_path: str):
    """
    生成详细的逐样本对比表格
    
    Args:
        all_results: 所有方法的结果列表
        output_path: 输出路径
    """
    # 创建 DataFrame
    data_rows = []
    
    for result in all_results:
        if result is None:
            continue
        
        method = result['method']
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        
        for i in range(len(result.get('total_cost', []))):
            row = {
                'Method': display_name,
                'Sample': i,
                'Total Cost': result['total_cost'][i] if result.get('total_cost') is not None else None,
                'Cost Ratio': result['cost_ratio'][i] if result.get('cost_ratio') is not None else None,
                'H-bar Distance': result['total_hbar_distance'][i] if result.get('total_hbar_distance') is not None else None,
                'Model Count': result['model_count'][i] if result.get('model_count') is not None else None,
                'Runtime': result['runtime'][i] if result.get('runtime') is not None else None,
            }
            data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    # 保存为 CSV
    csv_path = output_path.replace('.txt', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"已保存详细结果: {csv_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='多方法对比可视化脚本')
    parser.add_argument('--methods', nargs='+', default=['parsac', 'superansac', 'ransac', 'kmeans', 'gmm'],
                        help='要对比的方法列表')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='结果目录路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录路径')
    parser.add_argument('--run_eval', action='store_true',
                        help='是否运行评估（如果结果文件不存在）')
    parser.add_argument('--known_count', action='store_true',
                        help='是否使用已知的模型数量')
    
    args = parser.parse_args()
    
    # 设置路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = args.results_dir or os.path.join(project_root, 'results')
    output_dir = args.output_dir or os.path.join(project_root, 'results', 'figures')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("多方法对比分析")
    print("=" * 60)
    print(f"对比方法: {args.methods}")
    print(f"结果目录: {results_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 加载所有方法的结果
    all_results = []
    
    for method in args.methods:
        print(f"\n加载 {method} 的结果...")
        result = load_method_results(method, results_dir)
        
        if result is None and args.run_eval:
            print(f"  结果文件不存在，运行评估...")
            result = run_method_evaluation(method, project_root, args.known_count)
        
        if result is not None:
            print(f"  成功加载 {method}")
            all_results.append(result)
        else:
            print(f"  警告: 无法加载 {method} 的结果")
    
    if not all_results:
        print("错误: 没有可用的结果数据")
        return
    
    print(f"\n成功加载 {len(all_results)} 个方法的结果")
    
    # 生成图表
    print("\n生成对比图表...")
    
    # 多联图
    create_multi_panel_figure(
        all_results,
        os.path.join(output_dir, 'multi_panel_comparison.png')
    )
    
    # 箱线图
    create_box_plot_comparison(
        all_results,
        os.path.join(output_dir, 'boxplot_comparison.png')
    )
    
    # 单独的指标图
    metrics_to_plot = [
        ('total_cost', 'Total Cost Comparison', 'Sum of Min Distances'),
        ('cost_ratio', 'Cost Ratio Comparison', 'Cost / GT Cost'),
        ('total_hbar_distance', 'H-bar Distance Comparison', 'Sum of H-bar Distances'),
    ]
    
    for metric, title, ylabel in metrics_to_plot:
        create_comparison_bar_chart(
            all_results, metric, title, ylabel,
            os.path.join(output_dir, f'bar_{metric}.png')
        )
    
    # 生成表格
    print("\n生成对比表格...")
    generate_comparison_table(all_results, os.path.join(output_dir, 'comparison_table.txt'))
    generate_detailed_table(all_results, os.path.join(output_dir, 'detailed_results.txt'))
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
