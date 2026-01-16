#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 V2 数据集的可视化图表
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 非交互式后端

# 添加项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def load_results(results_dir, methods):
    """加载指定目录下所有方法的结果"""
    results = {}
    for method in methods:
        csv_path = os.path.join(results_dir, method, f"{method}_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            results[method] = df
    return results


def plot_bar_comparison(results, metric, title, ylabel, output_path, methods_order=None):
    """绘制柱状图对比"""
    if methods_order is None:
        methods_order = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(methods_order))
    width = 0.6
    
    means = []
    stds = []
    colors = []
    color_map = {
        'ours': '#2E86AB',
        'parsac_known': '#A23B72',
        'parsac_unknown': '#F18F01',
        'superansac_known': '#C73E1D',
        'superansac_unknown': '#3B1F2B',
        'ransac': '#95C623',
        'kmeans': '#7B2D8E',
        'gmm': '#009B77'
    }
    
    for method in methods_order:
        if method in results:
            means.append(results[method][metric].mean())
            stds.append(results[method][metric].std())
            colors.append(color_map.get(method, '#666666'))
        else:
            means.append(0)
            stds.append(0)
            colors.append('#CCCCCC')
    
    bars = ax.bar(x, means, width, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in methods_order], rotation=0, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.annotate(f'{mean:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")


def plot_v1_v2_comparison(results_v1, results_v2, metric, title, ylabel, output_path, methods_order=None):
    """绘制 V1 vs V2 对比柱状图"""
    if methods_order is None:
        methods_order = list(results_v1.keys())
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(methods_order))
    width = 0.35
    
    means_v1 = []
    means_v2 = []
    stds_v1 = []
    stds_v2 = []
    
    for method in methods_order:
        if method in results_v1:
            means_v1.append(results_v1[method][metric].mean())
            stds_v1.append(results_v1[method][metric].std())
        else:
            means_v1.append(0)
            stds_v1.append(0)
        
        if method in results_v2:
            means_v2.append(results_v2[method][metric].mean())
            stds_v2.append(results_v2[method][metric].std())
        else:
            means_v2.append(0)
            stds_v2.append(0)
    
    bars1 = ax.bar(x - width/2, means_v1, width, yerr=stds_v1, capsize=3, 
                   label='V1 (4hp, noise=0.1)', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, means_v2, width, yerr=stds_v2, capsize=3,
                   label='V2 (6hp, noise=0.2)', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in methods_order], rotation=0, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")


def main():
    methods = ['ours', 'parsac_known', 'parsac_unknown', 
               'superansac_known', 'superansac_unknown',
               'ransac', 'kmeans', 'gmm']
    
    # 加载数据
    results_2d_v1 = load_results(os.path.join(project_root, 'results', '2d'), methods)
    results_2d_v2 = load_results(os.path.join(project_root, 'results', '2d-2'), methods)
    results_3d_v1 = load_results(os.path.join(project_root, 'results', '3d'), methods)
    results_3d_v2 = load_results(os.path.join(project_root, 'results', '3d-2'), methods)
    
    # 2D 图表
    output_2d = os.path.join(project_root, 'results', '2d-2', 'figures')
    os.makedirs(output_2d, exist_ok=True)
    
    plot_bar_comparison(results_2d_v2, 'cost_ratio', 
                       '2D Line Fitting - Cost Ratio (V2 Dataset)', 'Cost Ratio',
                       os.path.join(output_2d, 'cost_ratio_2d_v2.png'), methods)
    
    plot_v1_v2_comparison(results_2d_v1, results_2d_v2, 'cost_ratio',
                         '2D Line Fitting - Cost Ratio Comparison (V1 vs V2)', 'Cost Ratio',
                         os.path.join(output_2d, 'cost_ratio_2d_v1_vs_v2.png'), methods)
    
    # 3D 图表
    output_3d = os.path.join(project_root, 'results', '3d-2', 'figures')
    os.makedirs(output_3d, exist_ok=True)
    
    plot_bar_comparison(results_3d_v2, 'cost_ratio',
                       '3D Plane Fitting - Cost Ratio (V2 Dataset)', 'Cost Ratio',
                       os.path.join(output_3d, 'cost_ratio_3d_v2.png'), methods)
    
    plot_v1_v2_comparison(results_3d_v1, results_3d_v2, 'cost_ratio',
                         '3D Plane Fitting - Cost Ratio Comparison (V1 vs V2)', 'Cost Ratio',
                         os.path.join(output_3d, 'cost_ratio_3d_v1_vs_v2.png'), methods)
    
    print("\n所有图表已生成完成！")


if __name__ == "__main__":
    main()
