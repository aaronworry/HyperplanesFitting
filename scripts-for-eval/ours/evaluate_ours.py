#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ours (流形优化) 方法评估脚本

评估我们的基于流形优化的超平面拟合算法在 2D 和 3D 数据集上的性能。

使用方法:
    # 2D 评估
    python scripts-for-eval/ours/evaluate_ours.py --dim 2 --known_count
    
    # 3D 评估
    python scripts-for-eval/ours/evaluate_ours.py --dim 3 --known_count

作者: Hyperplanes Fitting Team
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from data.read_data import read_data_2D, read_data_3D
from algorithm.initial_value import Hyperplane, Polyhedron
from algorithm.hyperplanes_fitting import HyperplanesFitting
from evaluate import evaluate


def evaluate_ours(data_dir: str, gt_dir: str, output_dir: str,
                  dim: int = 2, num_samples: int = 20,
                  known_count: bool = True,
                  method: str = "3",
                  use_initial: bool = True,
                  true_num: int = 4):
    """
    评估 Ours (流形优化) 方法
    
    Args:
        data_dir: 数据目录
        gt_dir: 真值目录
        output_dir: 输出目录
        dim: 数据维度 (2 或 3)
        num_samples: 样本数量
        known_count: 是否已知模型数量
        method: 优化方法 ("1"=A, "2"=B, "3"=A+B)
        use_initial: 是否使用初始值估计
        true_num: 已知的模型数量 (仅在 use_initial=False 时使用)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # 选择读取函数
    read_data_func = read_data_2D if dim == 2 else read_data_3D
    
    # 初始化算法
    alg = HyperplanesFitting(dim, None, parallel=False, method=method, whether_initial_value=use_initial)
    
    print(f"\n{'='*60}")
    print(f"Ours (流形优化) {dim}D 评估")
    print(f"{'='*60}")
    print(f"数据目录: {data_dir}")
    print(f"真值目录: {gt_dir}")
    print(f"输出目录: {output_dir}")
    print(f"已知模型数量: {known_count}")
    print(f"优化方法: {method}")
    print(f"使用初始值估计: {use_initial}")
    print(f"{'='*60}\n")
    
    for i in range(num_samples):
        try:
            # 读取数据
            data, gt_data, gt_total_cost = read_data_func(data_dir, gt_dir, file_index=i)
            
            # 构建真值
            gt_hyperplanes = []
            for j in range(len(gt_data)):
                gt_hyperplanes.append(Hyperplane(gt_data[j, :dim], gt_data[j, dim]))
            ground_truth_poly = Polyhedron(dim, gt_hyperplanes)
            
            # 设置数据
            alg.set_data(data)
            
            # 计时
            start_time = time.time()
            
            # 求解
            if use_initial or not known_count:
                hps = alg.solve(None)
            else:
                num_hp = len(gt_hyperplanes) if known_count else true_num
                hps = alg.solve(num_hp)
            
            runtime = time.time() - start_time
            
            # 构建结果
            result_poly = Polyhedron(dim, hps)
            
            # 评估
            total_hbar_distance, total_cost, average_distance, gt_avg_distance = evaluate(
                data, ground_truth_poly, gt_total_cost, result_poly
            )
            
            # 计算 cost ratio
            cost_ratio = total_cost / gt_total_cost if gt_total_cost > 0 else float('inf')
            
            # 模型数量
            model_count = len(hps)
            gt_model_count = len(gt_hyperplanes)
            model_count_error = abs(model_count - gt_model_count)
            
            results.append({
                'sample_id': i,
                'total_cost': total_cost,
                'cost_ratio': cost_ratio,
                'average_distance': average_distance,
                'total_hbar_distance': total_hbar_distance,
                'model_count': model_count,
                'gt_model_count': gt_model_count,
                'model_count_error': model_count_error,
                'runtime': runtime,
            })
            
            print(f"样本 {i:2d}: TC={total_cost:.4f}, Ratio={cost_ratio:.4f}, "
                  f"HE={total_hbar_distance:.4f}, Models={model_count}/{gt_model_count}, "
                  f"Time={runtime:.4f}s")
            
        except Exception as e:
            print(f"样本 {i} 处理失败: {e}")
            continue
    
    # 保存结果
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, 'ours_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n结果已保存到: {csv_path}")
        
        # 输出统计汇总
        print(f"\n{'='*60}")
        print("汇总统计")
        print(f"{'='*60}")
        print(f"平均模型数量: {df['model_count'].mean():.2f} ± {df['model_count'].std():.2f}")
        print(f"平均 Total Cost: {df['total_cost'].mean():.4f} ± {df['total_cost'].std():.4f}")
        print(f"平均 Cost Ratio: {df['cost_ratio'].mean():.4f} ± {df['cost_ratio'].std():.4f}")
        print(f"平均 Hbar Distance: {df['total_hbar_distance'].mean():.4f} ± {df['total_hbar_distance'].std():.4f}")
        print(f"平均 Runtime: {df['runtime'].mean():.4f}s ± {df['runtime'].std():.4f}s")
        
        # 保存汇总到文本文件
        summary_path = os.path.join(output_dir, 'ours_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Ours (流形优化) {dim}D 评估结果\n")
            f.write(f"{'='*40}\n")
            f.write(f"样本数量: {len(results)}\n")
            f.write(f"优化方法: {method}\n")
            f.write(f"使用初始值估计: {use_initial}\n\n")
            f.write(f"平均模型数量: {df['model_count'].mean():.2f} ± {df['model_count'].std():.2f}\n")
            f.write(f"平均 Total Cost: {df['total_cost'].mean():.4f} ± {df['total_cost'].std():.4f}\n")
            f.write(f"平均 Cost Ratio: {df['cost_ratio'].mean():.4f} ± {df['cost_ratio'].std():.4f}\n")
            f.write(f"平均 Hbar Distance: {df['total_hbar_distance'].mean():.4f} ± {df['total_hbar_distance'].std():.4f}\n")
            f.write(f"平均 Runtime: {df['runtime'].mean():.4f}s ± {df['runtime'].std():.4f}s\n")
        print(f"汇总已保存到: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Ours (流形优化) 方法评估')
    parser.add_argument('--dim', type=int, default=2, choices=[2, 3],
                        help='数据维度 (2 或 3)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录 (默认: csv_dataset 或 csv_dataset_3d)')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='真值目录 (默认: csv_groundtruth 或 csv_groundtruth_3d)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认: results/2d/ours 或 results/3d/ours)')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='评估的样本数量')
    parser.add_argument('--known_count', action='store_true',
                        help='使用已知的模型数量')
    parser.add_argument('--method', type=str, default='3', choices=['1', '2', '3'],
                        help='优化方法: 1=A, 2=B, 3=A+B')
    parser.add_argument('--no_initial', action='store_true',
                        help='不使用初始值估计')
    parser.add_argument('--true_num', type=int, default=4,
                        help='已知的模型数量 (仅在 --no_initial 时使用)')
    
    args = parser.parse_args()
    
    # 设置默认路径
    if args.data_dir is None:
        args.data_dir = 'csv_dataset' if args.dim == 2 else 'csv_dataset_3d'
    if args.gt_dir is None:
        args.gt_dir = 'csv_groundtruth' if args.dim == 2 else 'csv_groundtruth_3d'
    if args.output_dir is None:
        args.output_dir = f'results/{args.dim}d/ours'
    
    evaluate_ours(
        data_dir=args.data_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        dim=args.dim,
        num_samples=args.num_samples,
        known_count=args.known_count,
        method=args.method,
        use_initial=not args.no_initial,
        true_num=args.true_num
    )


if __name__ == '__main__':
    main()
