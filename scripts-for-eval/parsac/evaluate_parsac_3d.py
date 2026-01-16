#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PARSAC 3D 平面拟合评估脚本

在 csv_dataset_3d 上评估 SimplePARSACPlaneFitter，输出格式与其他对比方法一致
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

from data.read_data import read_data_3D
from plane_fitter_3d import SimplePARSACPlaneFitter


class Hyperplane3D:
    """3D 超平面类"""
    def __init__(self, normal, distance):
        self.normal = np.array(normal)
        norm = np.linalg.norm(self.normal)
        if norm > 1e-10:
            self.normal = self.normal / norm
        self.distance = distance
    
    def get_hbar(self):
        """返回 h-bar 向量: d * n"""
        return self.distance * self.normal


class Polyhedron3D:
    """3D 多面体（超平面集合）"""
    def __init__(self, hyperplanes):
        self.hyperplanes = hyperplanes
    
    def __len__(self):
        return len(self.hyperplanes)


def compute_total_cost(points: np.ndarray, hyperplanes: list) -> float:
    """计算点到最近超平面的总距离"""
    if len(hyperplanes) == 0:
        return float('inf')
    
    N = len(points)
    min_distances = np.full(N, np.inf)
    
    for hp in hyperplanes:
        # 点到平面的距离: |n · x - d|
        distances = np.abs(np.dot(points, hp.normal) - hp.distance)
        min_distances = np.minimum(min_distances, distances)
    
    return np.sum(min_distances)


def compute_hbar_distance(result_hps: list, gt_hps: list) -> float:
    """计算 H-bar 距离"""
    if len(result_hps) == 0:
        return float('inf')
    
    total_distance = 0.0
    
    for res_hp in result_hps:
        res_hbar = res_hp.get_hbar()
        min_dist = float('inf')
        
        for gt_hp in gt_hps:
            gt_hbar = gt_hp.get_hbar()
            dist = np.linalg.norm(res_hbar - gt_hbar)
            min_dist = min(min_dist, dist)
        
        total_distance += min_dist
    
    return total_distance


def run_parsac_3d_on_sample(data: np.ndarray, 
                            num_planes: int = None,
                            num_hypotheses: int = 300,
                            inlier_threshold: float = 0.2,
                            num_iterations: int = 3,
                            auto_detect: bool = False) -> tuple:
    """
    在单个 3D 样本上运行 PARSAC
    
    Args:
        data: 点云数据 (n x 3)
        num_planes: 拟合的平面数量，None 表示自动检测
        num_hypotheses: 假设数量
        inlier_threshold: 内点阈值
        num_iterations: 迭代次数
        auto_detect: 是否自动检测模型数量（当 num_planes=None 时生效）
        
    Returns:
        hyperplanes: 拟合的超平面列表
        runtime: 运行时间
    """
    fitter = SimplePARSACPlaneFitter(
        num_hypotheses=num_hypotheses,
        num_instances=num_planes or 4,
        inlier_threshold=inlier_threshold
    )
    
    start_time = time.time()
    planes, labels = fitter.fit(data, num_models=num_planes, 
                               num_iterations=num_iterations,
                               auto_detect=auto_detect)
    runtime = time.time() - start_time
    
    # 转换为 Hyperplane3D 对象
    hyperplanes = []
    for plane in planes:
        n = np.array([plane[0], plane[1], plane[2]])
        d = plane[3]
        hyperplanes.append(Hyperplane3D(normal=n, distance=d))
    
    return hyperplanes, runtime


def evaluate_dataset_3d(data_dir: str,
                        gt_dir: str,
                        num_samples: int = 20,
                        known_count: bool = False,
                        num_hypotheses: int = 300,
                        inlier_threshold: float = 0.2,
                        num_iterations: int = 3,
                        verbose: bool = True) -> list:
    """
    在整个 3D 数据集上评估 PARSAC
    """
    results = []
    
    for i in range(num_samples):
        if verbose:
            print(f"处理样本 {i}...")
        
        # 读取数据
        data, gt_data, gt_total_cost = read_data_3D(data_dir, gt_dir, file_index=i)
        
        # 构建真值超平面
        gt_hyperplanes = []
        for j in range(len(gt_data)):
            n = gt_data[j, :3]  # 法向量 (n1, n2, n3)
            d = gt_data[j, 3]   # 距离
            gt_hyperplanes.append(Hyperplane3D(normal=n, distance=d))
        
        # 确定拟合数量和自动检测标志
        if known_count:
            num_planes = len(gt_hyperplanes)
            auto_detect = False
        else:
            num_planes = None
            auto_detect = True
        
        # 运行 PARSAC
        hyperplanes, runtime = run_parsac_3d_on_sample(
            data, 
            num_planes=num_planes,
            num_hypotheses=num_hypotheses,
            inlier_threshold=inlier_threshold,
            num_iterations=num_iterations,
            auto_detect=auto_detect
        )
        
        # 评估
        total_cost = compute_total_cost(data, hyperplanes)
        cost_ratio = total_cost / gt_total_cost if gt_total_cost > 0 else float('inf')
        average_distance = total_cost / len(data)
        gt_average_distance = gt_total_cost / len(data)
        hbar_distance = compute_hbar_distance(hyperplanes, gt_hyperplanes)
        model_count = len(hyperplanes)
        gt_model_count = len(gt_hyperplanes)
        model_count_error = abs(model_count - gt_model_count)
        
        result = {
            'sample_id': i,
            'total_cost': total_cost,
            'cost_ratio': cost_ratio,
            'average_distance': average_distance,
            'gt_average_distance': gt_average_distance,
            'total_hbar_distance': hbar_distance,
            'model_count': model_count,
            'gt_model_count': gt_model_count,
            'model_count_error': model_count_error,
            'runtime': runtime
        }
        results.append(result)
        
        if verbose:
            print(f"\n=== Sample {i} ===")
            print(f"  Total Cost:          {total_cost:.4f}")
            print(f"  GT Total Cost:       {gt_total_cost:.4f}")
            print(f"  Cost Ratio:          {cost_ratio:.4f}")
            print(f"  Average Distance:    {average_distance:.6f}")
            print(f"  Hbar Distance:       {hbar_distance:.4f}")
            print(f"  Model Count:         {model_count} (GT: {gt_model_count})")
            print(f"  Runtime: {runtime:.4f}s")
    
    return results


def save_results(results: list, output_dir: str, method_name: str = "parsac_3d"):
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
        for r in results:
            writer.writerow({
                'sample_id': r['sample_id'],
                'total_cost': r['total_cost'],
                'cost_ratio': r['cost_ratio'],
                'average_distance': r['average_distance'],
                'total_hbar_distance': r['total_hbar_distance'],
                'model_count': r['model_count'],
                'gt_model_count': r['gt_model_count'],
                'model_count_error': r['model_count_error'],
                'runtime': r['runtime']
            })
    
    # 保存汇总统计
    summary_path = os.path.join(output_dir, f"{method_name}_summary.txt")
    
    total_costs = [r['total_cost'] for r in results]
    cost_ratios = [r['cost_ratio'] for r in results]
    hbar_distances = [r['total_hbar_distance'] for r in results]
    model_errors = [r['model_count_error'] for r in results]
    runtimes = [r['runtime'] for r in results]
    model_counts = [r['model_count'] for r in results]
    
    with open(summary_path, 'w') as f:
        f.write(f"=== {method_name.upper()} 3D 评估汇总 ===\n")
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
    total_costs = [r['total_cost'] for r in results]
    cost_ratios = [r['cost_ratio'] for r in results]
    hbar_distances = [r['total_hbar_distance'] for r in results]
    model_errors = [r['model_count_error'] for r in results]
    runtimes = [r['runtime'] for r in results]
    model_counts = [r['model_count'] for r in results]
    
    print("\n" + "=" * 50)
    print("PARSAC 3D 汇总统计")
    print("=" * 50)
    print(f"样本数: {len(results)}")
    print(f"平均模型数量: {np.mean(model_counts):.2f} ± {np.std(model_counts):.2f}")
    print(f"平均 Total Cost: {np.mean(total_costs):.4f} ± {np.std(total_costs):.4f}")
    print(f"平均 Cost Ratio: {np.mean(cost_ratios):.4f} ± {np.std(cost_ratios):.4f}")
    print(f"平均 Hbar Distance: {np.mean(hbar_distances):.4f} ± {np.std(hbar_distances):.4f}")
    print(f"平均 Model Count Error: {np.mean(model_errors):.2f} ± {np.std(model_errors):.2f}")
    print(f"平均 Runtime: {np.mean(runtimes):.4f}s ± {np.std(runtimes):.4f}s")


def main():
    parser = argparse.ArgumentParser(description='PARSAC 3D 平面拟合评估脚本')
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
    parser.add_argument('--num_hypotheses', type=int, default=300,
                        help='假设数量')
    parser.add_argument('--inlier_threshold', type=float, default=0.2,
                        help='内点阈值')
    parser.add_argument('--num_iterations', type=int, default=3,
                        help='迭代次数')
    
    args = parser.parse_args()
    
    # 设置默认路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = args.data_dir or os.path.join(project_root, 'csv_dataset_3d')
    gt_dir = args.gt_dir or os.path.join(project_root, 'csv_groundtruth_3d')
    output_dir = args.output_dir or os.path.join(project_root, 'results', '3d', 'parsac')
    
    print("=" * 50)
    print("PARSAC 3D 平面拟合评估")
    print("=" * 50)
    print(f"数据目录: {data_dir}")
    print(f"真值目录: {gt_dir}")
    print(f"已知模型数量: {args.known_count}")
    print(f"假设数量: {args.num_hypotheses}")
    print(f"内点阈值: {args.inlier_threshold}")
    print(f"迭代次数: {args.num_iterations}")
    print("=" * 50)
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"\n错误: 数据目录不存在: {data_dir}")
        print("请先运行以下命令生成 3D 数据:")
        print("  cd data && python csv_data_generator.py --dim 3")
        return
    
    # 运行评估
    results = evaluate_dataset_3d(
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
