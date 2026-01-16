#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D 多方法对比可视化脚本

此脚本用于:
1. 读取所有已评估方法在 3D 数据集上的结果
2. 运行各方法的 3D 评估 (如果结果不存在)
3. 生成多联对比图 (与论文中的图表格式一致)
4. 生成数据表格并保存到 txt 文件

使用方法:
    python compare_all_methods_3d.py --methods ours ransac kmeans gmm parsac superansac
    python compare_all_methods_3d.py --run_eval --known_count

作者: Hyperplanes Fitting Team
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))

from data.read_data import read_data_3D
from algorithm.initial_value import Hyperplane, Polyhedron
from evaluate import evaluate

# ============================================================================
# 配置区域
# ============================================================================

AVAILABLE_METHODS = [
    'ours',
    'parsac',
    'superansac',
    'ransac',
    'kmeans',
    'gmm',
]

METHOD_DISPLAY_NAMES = {
    'ours': 'Ours (Manifold Opt.)',
    'parsac': 'PARSAC',
    'superansac': 'SupeRANSAC',
    'ransac': 'RANSAC',
    'kmeans': 'K-Means',
    'gmm': 'GMM',
}

METHOD_COLORS = {
    'ours': '#e41a1c',
    'parsac': '#8dd3c7',
    'superansac': '#fb8072',
    'ransac': '#377eb8',
    'kmeans': '#4daf4a',
    'gmm': '#984ea3',
}

# ============================================================================
# 数据加载函数
# ============================================================================

def load_method_results(method: str, results_dir: str) -> dict:
    """加载指定方法的评估结果"""
    # 尝试不同的文件名格式
    possible_names = [
        f"{method}_results.csv",
        f"{method}_3d_results.csv",
    ]
    
    for filename in possible_names:
        csv_path = os.path.join(results_dir, method, filename)
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
                print(f"警告: 无法加载 {method} 的结果 ({filename}): {e}")
    return None


def run_method_evaluation(method: str, data_dir: str, gt_dir: str, 
                          output_dir: str, num_samples: int = 20,
                          known_count: bool = True) -> dict:
    """运行指定方法的 3D 评估"""
    
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
    
    method_output_dir = os.path.join(output_dir, method)
    os.makedirs(method_output_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"评估方法: {METHOD_DISPLAY_NAMES.get(method, method)} (3D)")
    print(f"{'='*50}")
    
    for i in range(num_samples):
        try:
            data, gt_data, gt_total_cost = read_data_3D(data_dir, gt_dir, file_index=i)
            
            # 构建真值
            gt_hyperplanes = []
            for j in range(len(gt_data)):
                gt_hyperplanes.append(Hyperplane(normal=gt_data[j, :3], distance=gt_data[j, 3]))
            ground_truth = Polyhedron(dim=3, hps=gt_hyperplanes)
            
            num_planes = len(gt_hyperplanes) if known_count else None
            
            # 根据方法选择拟合器
            if method == 'ours':
                from algorithm.hyperplanes_fitting import HyperplanesFitting
                alg = HyperplanesFitting(3, data, parallel=False, method="3", whether_initial_value=True)
                start = time.time()
                hps = alg.solve(None)
                runtime = time.time() - start
                hyperplanes = hps
                
            elif method == 'parsac':
                sys.path.insert(0, os.path.join(project_root, 'scripts-for-eval'))
                from parsac.plane_fitter_3d import SimplePARSACPlaneFitter
                fitter = SimplePARSACPlaneFitter(num_hypotheses=500, inlier_threshold=0.2)
                start = time.time()
                planes = fitter.fit(data, num_planes=num_planes)
                runtime = time.time() - start
                hyperplanes = [Hyperplane(normal=n, distance=d) for n, d in planes]
                
            elif method == 'superansac':
                sys.path.insert(0, os.path.join(project_root, 'scripts-for-eval'))
                from superansac.sequential_ransac_3d import SequentialRANSAC3DPlane, RANSACConfig3D
                config = RANSACConfig3D(max_iterations=1000, inlier_threshold=0.3, min_inliers=15)
                fitter = SequentialRANSAC3DPlane(config=config)
                start = time.time()
                planes = fitter.fit(data, max_planes=num_planes)
                runtime = time.time() - start
                hyperplanes = [Hyperplane(normal=p.normal, distance=p.distance) for p in planes]
                
            elif method == 'ransac':
                sys.path.insert(0, os.path.join(project_root, 'scripts-for-eval'))
                from compared_alg_3d.RANSAC_3D import RANSAC3D
                alg = RANSAC3D(num_planes or 4, 3)
                alg.set_data(data)
                start = time.time()
                alg.solve()
                runtime = time.time() - start
                hyperplanes = [Hyperplane(normal=alg.vectors[k], distance=alg.distances[k])
                              for k in range(alg.n)]
                
            elif method == 'kmeans':
                sys.path.insert(0, os.path.join(project_root, 'scripts-for-eval'))
                from compared_alg_3d.K_Means_3D import KMeans3D
                alg = KMeans3D(num_planes or 4, 3)
                alg.set_data(data)
                start = time.time()
                alg.solve()
                runtime = time.time() - start
                hyperplanes = [Hyperplane(normal=alg.vectors[k], distance=alg.distances[k])
                              for k in range(alg.n)]
                
            elif method == 'gmm':
                sys.path.insert(0, os.path.join(project_root, 'scripts-for-eval'))
                from compared_alg_3d.GMM_3D import GMM3D
                alg = GMM3D(num_planes or 4, 3)
                alg.set_data(data)
                start = time.time()
                alg.solve()
                runtime = time.time() - start
                hyperplanes = [Hyperplane(normal=alg.vectors[k], distance=alg.distances[k])
                              for k in range(alg.n)]
            else:
                print(f"未知方法: {method}")
                continue
            
            # 评估
            result = Polyhedron(dim=3, hps=hyperplanes)
            total_hbar_distance, total_cost, average_distance, gt_avg_distance = evaluate(
                data, ground_truth, gt_total_cost, result
            )
            
            cost_ratio = total_cost / gt_total_cost if gt_total_cost > 0 else float('inf')
            model_count = len(hyperplanes)
            gt_model_count = len(gt_hyperplanes)
            model_count_error = abs(model_count - gt_model_count)
            
            results['total_cost'].append(total_cost)
            results['cost_ratio'].append(cost_ratio)
            results['average_distance'].append(average_distance)
            results['total_hbar_distance'].append(total_hbar_distance)
            results['model_count'].append(model_count)
            results['model_count_error'].append(model_count_error)
            results['runtime'].append(runtime)
            
            print(f"  样本 {i:2d}: TC={total_cost:.4f}, Ratio={cost_ratio:.4f}, "
                  f"HE={total_hbar_distance:.4f}, Time={runtime:.4f}s")
            
        except Exception as e:
            print(f"  样本 {i} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 转换为 numpy 数组
    for key in results:
        if key != 'method' and results[key]:
            results[key] = np.array(results[key])
    
    # 保存结果
    if results['total_cost'] is not None and len(results['total_cost']) > 0:
        df = pd.DataFrame({
            'sample_id': range(len(results['total_cost'])),
            'total_cost': results['total_cost'],
            'cost_ratio': results['cost_ratio'],
            'average_distance': results['average_distance'],
            'total_hbar_distance': results['total_hbar_distance'],
            'model_count': results['model_count'],
            'model_count_error': results['model_count_error'],
            'runtime': results['runtime'],
        })
        df['gt_model_count'] = len(gt_hyperplanes)
        csv_path = os.path.join(method_output_dir, f"{method}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"  结果已保存到: {csv_path}")
    
    return results


# ============================================================================
# 可视化函数
# ============================================================================

def create_multi_panel_figure(all_results: list, output_path: str):
    """创建多联图"""
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
    
    plt.suptitle('Multi-Method Comparison on 3D Plane Fitting Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存多联图: {output_path}")


def create_box_plot_comparison(all_results: list, output_path: str):
    """创建箱线图对比"""
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
        colors = []
        
        for result in all_results:
            if result is None or result.get(metric) is None:
                continue
            method = result['method']
            data = result[metric]
            if data is None or len(data) == 0:
                continue
            data_to_plot.append(data)
            labels.append(METHOD_DISPLAY_NAMES.get(method, method))
            colors.append(METHOD_COLORS.get(method, '#333333'))
        
        if not data_to_plot:
            continue
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        for box, color in zip(bp['boxes'], colors):
            box.set_facecolor(color)
            box.set_alpha(0.7)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('3D Plane Fitting: Box Plot Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存箱线图: {output_path}")


def create_comparison_table(all_results: list, output_path: str):
    """创建对比表格"""
    lines = []
    lines.append("=" * 120)
    lines.append("Multi-Method Comparison Results (3D Plane Fitting)")
    lines.append("=" * 120)
    lines.append("")
    
    header = f"{'Method':^14} | {'Model Count':^15} | {'Total Cost':^15} | {'Cost Ratio':^15} | {'H-bar Distance':^15} | {'Avg Distance':^17} | {'Runtime (s)':^14}"
    lines.append(header)
    lines.append("-" * 120)
    
    for result in all_results:
        if result is None:
            continue
        method = result['method']
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        
        def format_stat(arr):
            if arr is None or len(arr) == 0:
                return "N/A"
            return f"{np.mean(arr):.4f} ± {np.std(arr):.4f}"
        
        line = f"{display_name:^14} | {format_stat(result.get('model_count')):^15} | " \
               f"{format_stat(result.get('total_cost')):^15} | {format_stat(result.get('cost_ratio')):^15} | " \
               f"{format_stat(result.get('total_hbar_distance')):^15} | {format_stat(result.get('average_distance')):^17} | " \
               f"{format_stat(result.get('runtime')):^14}"
        lines.append(line)
    
    lines.append("-" * 120)
    lines.append("")
    
    # LaTeX 表格
    lines.append("=" * 80)
    lines.append("LaTeX Table Format:")
    lines.append("=" * 80)
    lines.append("")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Comparison of Different Methods on 3D Plane Fitting Dataset}")
    lines.append("\\label{tab:comparison_3d}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Method & Model Count & Total Cost & Cost Ratio & H-bar Dist. & Avg Dist. & Runtime (s) \\\\")
    lines.append("\\midrule")
    
    for result in all_results:
        if result is None:
            continue
        method = result['method']
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        
        def format_latex(arr):
            if arr is None or len(arr) == 0:
                return "N/A"
            return f"${np.mean(arr):.3f}$"
        
        line = f"{display_name} & {format_latex(result.get('model_count'))} & " \
               f"{format_latex(result.get('total_cost'))} & {format_latex(result.get('cost_ratio'))} & " \
               f"{format_latex(result.get('total_hbar_distance'))} & {format_latex(result.get('average_distance'))} & " \
               f"{format_latex(result.get('runtime'))} \\\\"
        lines.append(line)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"已保存对比表格: {output_path}")
    print('\n'.join(lines[:20]))


def save_detailed_results(all_results: list, output_path: str):
    """保存详细结果到 CSV"""
    rows = []
    for result in all_results:
        if result is None:
            continue
        method = result['method']
        for i in range(len(result.get('total_cost', []))):
            row = {
                'method': method,
                'sample_id': i,
                'total_cost': result['total_cost'][i] if result.get('total_cost') is not None else None,
                'cost_ratio': result['cost_ratio'][i] if result.get('cost_ratio') is not None else None,
                'average_distance': result['average_distance'][i] if result.get('average_distance') is not None else None,
                'total_hbar_distance': result['total_hbar_distance'][i] if result.get('total_hbar_distance') is not None else None,
                'model_count': result['model_count'][i] if result.get('model_count') is not None else None,
                'runtime': result['runtime'][i] if result.get('runtime') is not None else None,
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"已保存详细结果: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='3D 多方法对比评估')
    parser.add_argument('--methods', nargs='+', default=['ours', 'parsac', 'superansac', 'ransac', 'kmeans', 'gmm'],
                        help='要对比的方法列表')
    parser.add_argument('--data_dir', type=str, default='csv_dataset_3d',
                        help='3D 数据目录')
    parser.add_argument('--gt_dir', type=str, default='csv_groundtruth_3d',
                        help='3D 真值目录')
    parser.add_argument('--results_dir', type=str, default='results/3d',
                        help='结果目录')
    parser.add_argument('--output_dir', type=str, default='results/3d/figures',
                        help='图表输出目录')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='样本数量')
    parser.add_argument('--run_eval', action='store_true',
                        help='运行评估 (如果结果不存在)')
    parser.add_argument('--force_eval', action='store_true',
                        help='强制重新评估所有方法')
    parser.add_argument('--known_count', action='store_true',
                        help='使用已知的模型数量')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    
    for method in args.methods:
        if method not in AVAILABLE_METHODS:
            print(f"警告: 方法 {method} 不可用，跳过")
            continue
        
        # 尝试加载已有结果
        result = None
        if not args.force_eval:
            result = load_method_results(method, args.results_dir)
        
        # 如果没有结果且允许运行评估
        if result is None and (args.run_eval or args.force_eval):
            result = run_method_evaluation(
                method, args.data_dir, args.gt_dir, args.results_dir,
                args.num_samples, args.known_count
            )
        
        if result is not None:
            all_results.append(result)
        else:
            print(f"警告: 方法 {method} 没有结果，跳过")
    
    if not all_results:
        print("错误: 没有可用的结果进行对比")
        return
    
    # 生成可视化
    print(f"\n{'='*50}")
    print("生成可视化图表")
    print(f"{'='*50}")
    
    create_multi_panel_figure(all_results, os.path.join(args.output_dir, 'multi_panel_comparison_3d.png'))
    create_box_plot_comparison(all_results, os.path.join(args.output_dir, 'boxplot_comparison_3d.png'))
    create_comparison_table(all_results, os.path.join(args.output_dir, 'comparison_table_3d.txt'))
    save_detailed_results(all_results, os.path.join(args.output_dir, 'detailed_results_3d.csv'))
    
    print(f"\n{'='*50}")
    print("3D 对比评估完成!")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
