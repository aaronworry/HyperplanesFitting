#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PARSAC 评估脚本 - 使用与论文完全一致的评估指标

在 csv_dataset 上评估 SimplePARSACLineFitter，输出格式与其他对比方法一致
"""

import sys
import os
import time
import argparse
import numpy as np

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'scripts-for-eval'))
sys.path.insert(0, os.path.dirname(__file__))

from data.read_data import read_data_2D
from evaluate_utils import Hyperplane, Polyhedron, full_evaluate, EvaluationResult
from line_fitter import SimplePARSACLineFitter


def run_parsac_on_sample(data: np.ndarray, 
                         num_lines: int = None,
                         num_hypotheses: int = 500,
                         inlier_threshold: float = 0.15,
                         num_iterations: int = 3) -> tuple:
    """
    在单个样本上运行 PARSAC
    
    Args:
        data: 点云数据 (n x 2)
        num_lines: 拟合的直线数量，None 表示自动检测
        num_hypotheses: 假设数量
        inlier_threshold: 内点阈值
        num_iterations: 迭代次数
        
    Returns:
        hyperplanes: 拟合的超平面列表
        runtime: 运行时间
    """
    fitter = SimplePARSACLineFitter(
        num_hypotheses=num_hypotheses,
        num_instances=num_lines or 4,
        inlier_threshold=inlier_threshold
    )
    
    start_time = time.time()
    lines, labels = fitter.fit(data, num_models=num_lines, num_iterations=num_iterations)
    runtime = time.time() - start_time
    
    # 转换为 Hyperplane 对象
    hyperplanes = []
    for line in lines:
        n = np.array([line[0], line[1]])
        d = line[2]
        hyperplanes.append(Hyperplane(normal=n, distance=d))
    
    return hyperplanes, runtime


def evaluate_dataset(data_dir: str,
                     gt_dir: str,
                     num_samples: int = 20,
                     known_count: bool = False,
                     num_hypotheses: int = 500,
                     inlier_threshold: float = 0.15,
                     num_iterations: int = 3,
                     verbose: bool = True) -> list:
    """
    在整个数据集上评估 PARSAC
    
    Args:
        data_dir: 数据目录
        gt_dir: 真值目录
        num_samples: 样本数量
        known_count: 是否已知模型数量
        num_hypotheses: 假设数量
        inlier_threshold: 内点阈值
        num_iterations: 迭代次数
        verbose: 是否打印详细信息
        
    Returns:
        results: 评估结果列表
    """
    results = []
    
    for i in range(num_samples):
        if verbose:
            print(f"处理样本 {i}...")
        
        # 读取数据
        data, gt_data, gt_total_cost = read_data_2D(data_dir, gt_dir, file_index=i)
        
        # 构建真值超平面
        gt_hyperplanes = []
        for j in range(len(gt_data)):
            n = gt_data[j, :2]  # 法向量
            d = gt_data[j, 2]   # 距离
            gt_hyperplanes.append(Hyperplane(normal=n, distance=d))
        ground_truth = Polyhedron(dim=2, hyperplanes=gt_hyperplanes)
        
        # 确定拟合数量
        num_lines = len(gt_hyperplanes) if known_count else None
        
        # 运行 PARSAC
        hyperplanes, runtime = run_parsac_on_sample(
            data, 
            num_lines=num_lines,
            num_hypotheses=num_hypotheses,
            inlier_threshold=inlier_threshold,
            num_iterations=num_iterations
        )
        
        # 构建结果多面体
        result = Polyhedron(dim=2, hyperplanes=hyperplanes)
        
        # 评估
        eval_result = full_evaluate(data, ground_truth, gt_total_cost, result, runtime)
        results.append(eval_result)
        
        if verbose:
            print(f"\n=== Sample {i} ===")
            print(f"  Total Cost:          {eval_result.total_cost:.4f}")
            print(f"  GT Total Cost:       {gt_total_cost:.4f}")
            print(f"  Cost Ratio:          {eval_result.cost_ratio:.4f}")
            print(f"  Average Distance:    {eval_result.average_distance:.6f}")
            print(f"  GT Average Distance: {eval_result.ground_truth_average_distance:.6f}")
            print(f"  Hbar Distance:       {eval_result.total_hbar_distance:.4f}")
            print(f"  Model Count:         {eval_result.model_count} (GT: {eval_result.gt_model_count})")
            print(f"  Model Count Error:   {eval_result.model_count_error}")
            print(f"  Runtime: {eval_result.runtime:.4f}s")
    
    return results


def save_results(results: list, output_dir: str, method_name: str = "parsac"):
    """保存评估结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细结果
    import csv
    csv_path = os.path.join(output_dir, f"{method_name}_results.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sample_id', 'total_cost', 'cost_ratio', 'average_distance',
            'total_hbar_distance', 'model_count', 'gt_model_count',
            'model_count_error', 'runtime'
        ])
        writer.writeheader()
        for i, r in enumerate(results):
            writer.writerow({
                'sample_id': i,
                'total_cost': r.total_cost,
                'cost_ratio': r.cost_ratio,
                'average_distance': r.average_distance,
                'total_hbar_distance': r.total_hbar_distance,
                'model_count': r.model_count,
                'gt_model_count': r.gt_model_count,
                'model_count_error': r.model_count_error,
                'runtime': r.runtime
            })
    
    # 保存汇总统计
    summary_path = os.path.join(output_dir, f"{method_name}_summary.txt")
    
    total_costs = [r.total_cost for r in results]
    cost_ratios = [r.cost_ratio for r in results]
    hbar_distances = [r.total_hbar_distance for r in results]
    model_errors = [r.model_count_error for r in results]
    runtimes = [r.runtime for r in results]
    model_counts = [r.model_count for r in results]
    
    with open(summary_path, 'w') as f:
        f.write(f"=== {method_name.upper()} 评估汇总 ===\n")
        f.write(f"样本数: {len(results)}\n\n")
        f.write(f"平均模型数量: {np.mean(model_counts):.2f} ± {np.std(model_counts):.2f}\n")
        f.write(f"平均 Total Cost: {np.mean(total_costs):.4f} ± {np.std(total_costs):.4f}\n")
        f.write(f"平均 Cost Ratio: {np.mean(cost_ratios):.4f} ± {np.std(cost_ratios):.4f}\n")
        f.write(f"平均 Hbar Distance: {np.mean(hbar_distances):.4f} ± {np.std(hbar_distances):.4f}\n")
        f.write(f"平均 Model Count Error: {np.mean(model_errors):.2f} ± {np.std(model_errors):.2f}\n")
        f.write(f"平均 Runtime: {np.mean(runtimes):.4f}s ± {np.std(runtimes):.4f}s\n")
    
    print(f"\n结果已保存到: {output_dir}")


def print_summary(results: list):
    """打印汇总统计"""
    total_costs = [r.total_cost for r in results]
    cost_ratios = [r.cost_ratio for r in results]
    hbar_distances = [r.total_hbar_distance for r in results]
    model_errors = [r.model_count_error for r in results]
    runtimes = [r.runtime for r in results]
    model_counts = [r.model_count for r in results]
    
    print("\n" + "=" * 50)
    print("汇总统计")
    print("=" * 50)
    print(f"样本数: {len(results)}")
    print(f"平均模型数量: {np.mean(model_counts):.2f} ± {np.std(model_counts):.2f}")
    print(f"平均 Total Cost: {np.mean(total_costs):.4f} ± {np.std(total_costs):.4f}")
    print(f"平均 Cost Ratio: {np.mean(cost_ratios):.4f} ± {np.std(cost_ratios):.4f}")
    print(f"平均 Hbar Distance: {np.mean(hbar_distances):.4f} ± {np.std(hbar_distances):.4f}")
    print(f"平均 Model Count Error: {np.mean(model_errors):.2f} ± {np.std(model_errors):.2f}")
    print(f"平均 Runtime: {np.mean(runtimes):.4f}s ± {np.std(runtimes):.4f}s")


def main():
    parser = argparse.ArgumentParser(description='PARSAC 评估脚本')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录路径')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='真值目录路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录路径')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='样本数量')
    parser.add_argument('--known_count', action='store_true',
                        help='是否使用已知的模型数量')
    parser.add_argument('--num_hypotheses', type=int, default=500,
                        help='假设数量')
    parser.add_argument('--inlier_threshold', type=float, default=0.15,
                        help='内点阈值')
    parser.add_argument('--num_iterations', type=int, default=3,
                        help='迭代次数')
    
    args = parser.parse_args()
    
    # 设置默认路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = args.data_dir or os.path.join(project_root, 'csv_dataset')
    gt_dir = args.gt_dir or os.path.join(project_root, 'csv_groundtruth')
    output_dir = args.output_dir or os.path.join(project_root, 'results', 'parsac')
    
    print("=" * 50)
    print("PARSAC 评估")
    print("=" * 50)
    print(f"数据目录: {data_dir}")
    print(f"真值目录: {gt_dir}")
    print(f"已知模型数量: {args.known_count}")
    print(f"假设数量: {args.num_hypotheses}")
    print(f"内点阈值: {args.inlier_threshold}")
    print(f"迭代次数: {args.num_iterations}")
    print("=" * 50)
    
    # 运行评估
    results = evaluate_dataset(
        data_dir=data_dir,
        gt_dir=gt_dir,
        num_samples=args.num_samples,
        known_count=args.known_count,
        num_hypotheses=args.num_hypotheses,
        inlier_threshold=args.inlier_threshold,
        num_iterations=args.num_iterations
    )
    
    # 打印和保存结果
    print_summary(results)
    save_results(results, output_dir)


if __name__ == "__main__":
    main()
