#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比两个数据集的评估结果

比较 2d vs 2d-2 和 3d vs 3d-2 的结果
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

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
            results[method] = {
                'total_cost': df['total_cost'].values,
                'cost_ratio': df['cost_ratio'].values,
                'total_hbar_distance': df['total_hbar_distance'].values,
                'model_count': df['model_count'].values,
                'gt_model_count': df['gt_model_count'].values,
                'model_count_error': df['model_count_error'].values,
                'runtime': df['runtime'].values
            }
        else:
            print(f"警告: 未找到 {csv_path}")
    return results


def format_value(mean, std, precision=2):
    """格式化均值±标准差"""
    return f"{mean:.{precision}f}±{std:.{precision}f}"


def compare_datasets(dim, results_v1, results_v2, methods, output_dir):
    """对比两个版本的结果"""
    
    # 创建对比表格
    comparison_table = []
    
    for method in methods:
        if method not in results_v1 or method not in results_v2:
            continue
        
        v1 = results_v1[method]
        v2 = results_v2[method]
        
        row = {
            'Method': method,
            # V1 数据集结果
            'V1_TC': format_value(np.mean(v1['total_cost']), np.std(v1['total_cost'])),
            'V1_CR': format_value(np.mean(v1['cost_ratio']), np.std(v1['cost_ratio'])),
            'V1_Models': format_value(np.mean(v1['model_count']), np.std(v1['model_count']), 1),
            'V1_Time': format_value(np.mean(v1['runtime']), np.std(v1['runtime']), 3) + 's',
            # V2 数据集结果
            'V2_TC': format_value(np.mean(v2['total_cost']), np.std(v2['total_cost'])),
            'V2_CR': format_value(np.mean(v2['cost_ratio']), np.std(v2['cost_ratio'])),
            'V2_Models': format_value(np.mean(v2['model_count']), np.std(v2['model_count']), 1),
            'V2_Time': format_value(np.mean(v2['runtime']), np.std(v2['runtime']), 3) + 's',
            # 改善百分比 (Cost Ratio)
            'CR_Change': f"{((np.mean(v2['cost_ratio']) - np.mean(v1['cost_ratio'])) / np.mean(v1['cost_ratio']) * 100):+.1f}%"
        }
        comparison_table.append(row)
    
    df = pd.DataFrame(comparison_table)
    
    # 保存 CSV
    csv_path = os.path.join(output_dir, f"comparison_{dim}d_v1_vs_v2.csv")
    df.to_csv(csv_path, index=False)
    
    # 生成 Markdown 表格
    md_path = os.path.join(output_dir, f"comparison_{dim}d_v1_vs_v2.md")
    with open(md_path, 'w') as f:
        f.write(f"# {dim}D 数据集对比 (V1 vs V2)\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 数据集说明\n\n")
        f.write("- **V1 (原始数据集)**: 4个超平面, 噪声=0.1, 每平面30点\n")
        f.write("- **V2 (新数据集)**: 6个超平面, 噪声=0.2, 每平面20-50点\n\n")
        
        f.write("## 结果对比\n\n")
        f.write("| Method | V1 CR | V2 CR | CR变化 | V1 Models | V2 Models | V1 Time | V2 Time |\n")
        f.write("|--------|-------|-------|--------|-----------|-----------|---------|--------|\n")
        
        for row in comparison_table:
            f.write(f"| {row['Method']} | {row['V1_CR']} | {row['V2_CR']} | {row['CR_Change']} | "
                   f"{row['V1_Models']} | {row['V2_Models']} | {row['V1_Time']} | {row['V2_Time']} |\n")
        
        f.write("\n## 关键发现\n\n")
        
        # 找出在 V2 上表现最好的方法
        best_v2_cr = min(comparison_table, key=lambda x: float(x['V2_CR'].split('±')[0]))
        f.write(f"- **V2 最佳方法 (Cost Ratio)**: {best_v2_cr['Method']} (CR={best_v2_cr['V2_CR']})\n")
        
        # 找出改善最大的方法
        def parse_change(s):
            return float(s.replace('%', '').replace('+', ''))
        
        # 找出 ours 的表现
        ours_row = next((r for r in comparison_table if r['Method'] == 'ours'), None)
        if ours_row:
            f.write(f"- **Ours 方法**: V1 CR={ours_row['V1_CR']}, V2 CR={ours_row['V2_CR']} (变化: {ours_row['CR_Change']})\n")
    
    print(f"已保存: {csv_path}")
    print(f"已保存: {md_path}")
    
    return df


def generate_summary_table(dim, results, methods, output_dir, dataset_name):
    """为单个数据集生成汇总表格"""
    
    summary = []
    for method in methods:
        if method not in results:
            continue
        
        r = results[method]
        summary.append({
            'Method': method,
            'Total Cost': format_value(np.mean(r['total_cost']), np.std(r['total_cost'])),
            'Cost Ratio': format_value(np.mean(r['cost_ratio']), np.std(r['cost_ratio'])),
            'H-bar Dist': format_value(np.mean(r['total_hbar_distance']), np.std(r['total_hbar_distance'])),
            'Models': format_value(np.mean(r['model_count']), np.std(r['model_count']), 1),
            'GT Models': r['gt_model_count'][0],
            'Model Err': format_value(np.mean(r['model_count_error']), np.std(r['model_count_error']), 1),
            'Runtime': format_value(np.mean(r['runtime']), np.std(r['runtime']), 3) + 's'
        })
    
    df = pd.DataFrame(summary)
    csv_path = os.path.join(output_dir, f"summary_{dim}d_{dataset_name}.csv")
    df.to_csv(csv_path, index=False)
    
    # 保存 TXT 格式
    txt_path = os.path.join(output_dir, f"comparison_table_{dim}d_{dataset_name}.txt")
    with open(txt_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"Multi-Method Comparison Results ({dim}D {dataset_name.upper()})\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'Method':<20} | {'Cost Ratio':<15} | {'H-bar Dist':<15} | {'Models':<12} | {'Runtime':<12}\n")
        f.write("-"*80 + "\n")
        
        for row in summary:
            f.write(f"{row['Method']:<20} | {row['Cost Ratio']:<15} | {row['H-bar Dist']:<15} | "
                   f"{row['Models']:<12} | {row['Runtime']:<12}\n")
    
    print(f"已保存: {csv_path}")
    print(f"已保存: {txt_path}")
    
    return df


def main():
    methods = ['ours', 'parsac_known', 'parsac_unknown', 
               'superansac_known', 'superansac_unknown',
               'ransac', 'kmeans', 'gmm']
    
    # 2D 对比
    print("\n" + "="*60)
    print("2D 数据集对比")
    print("="*60)
    
    results_2d_v1 = load_results(os.path.join(project_root, 'results', '2d'), methods)
    results_2d_v2 = load_results(os.path.join(project_root, 'results', '2d-2'), methods)
    
    output_2d = os.path.join(project_root, 'results', '2d-2', 'figures')
    os.makedirs(output_2d, exist_ok=True)
    
    compare_datasets(2, results_2d_v1, results_2d_v2, methods, output_2d)
    generate_summary_table(2, results_2d_v2, methods, output_2d, 'v2')
    
    # 3D 对比
    print("\n" + "="*60)
    print("3D 数据集对比")
    print("="*60)
    
    results_3d_v1 = load_results(os.path.join(project_root, 'results', '3d'), methods)
    results_3d_v2 = load_results(os.path.join(project_root, 'results', '3d-2'), methods)
    
    output_3d = os.path.join(project_root, 'results', '3d-2', 'figures')
    os.makedirs(output_3d, exist_ok=True)
    
    compare_datasets(3, results_3d_v1, results_3d_v2, methods, output_3d)
    generate_summary_table(3, results_3d_v2, methods, output_3d, 'v2')
    
    # 打印最终对比结果
    print("\n" + "="*60)
    print("最终对比汇总")
    print("="*60)
    
    print("\n2D Cost Ratio 对比:")
    print(f"{'Method':<20} {'V1 (4hp, 0.1noise)':<20} {'V2 (6hp, 0.2noise)':<20}")
    print("-"*60)
    for method in methods:
        if method in results_2d_v1 and method in results_2d_v2:
            v1_cr = np.mean(results_2d_v1[method]['cost_ratio'])
            v2_cr = np.mean(results_2d_v2[method]['cost_ratio'])
            print(f"{method:<20} {v1_cr:.4f}                 {v2_cr:.4f}")
    
    print("\n3D Cost Ratio 对比:")
    print(f"{'Method':<20} {'V1 (4hp, 0.1noise)':<20} {'V2 (6hp, 0.2noise)':<20}")
    print("-"*60)
    for method in methods:
        if method in results_3d_v1 and method in results_3d_v2:
            v1_cr = np.mean(results_3d_v1[method]['cost_ratio'])
            v2_cr = np.mean(results_3d_v2[method]['cost_ratio'])
            print(f"{method:<20} {v1_cr:.4f}                 {v2_cr:.4f}")


if __name__ == "__main__":
    main()
