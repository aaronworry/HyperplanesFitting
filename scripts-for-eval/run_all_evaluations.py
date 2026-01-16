#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的多方法评估脚本

此脚本运行所有方法的评估，包括:
- Ours (流形优化方法)
- PARSAC (已知/未知模型数量)
- SupeRANSAC (已知/未知模型数量)
- RANSAC
- K-Means
- GMM

支持 2D 直线和 3D 平面拟合

使用方法:
    python run_all_evaluations.py --dim 2    # 运行 2D 评估
    python run_all_evaluations.py --dim 3    # 运行 3D 评估
    python run_all_evaluations.py --dim all  # 运行所有评估
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))


def get_2d_evaluators():
    """获取 2D 评估函数字典"""
    evaluators = {}
    
    # Ours (流形优化方法)
    try:
        from algorithm.hyperplanes_fitting import HyperplanesFitting
        from algorithm.initial_value import Hyperplane as AlgHyperplane, Polyhedron as AlgPolyhedron
        from evaluate_utils import Hyperplane
        
        def eval_ours(data, gt_data, gt_total_cost, num_models):
            # 使用初始值估计（自动检测超平面数量）
            alg = HyperplanesFitting(dim=2, data=data, parallel=False, 
                                     method="3", whether_initial_value=True)
            start = time.time()
            hps = alg.solve(true_num=None)
            runtime = time.time() - start
            
            hyperplanes = []
            for hp in hps:
                hyperplanes.append(Hyperplane(normal=hp.normal, distance=hp.distance))
            return hyperplanes, runtime
        
        evaluators['ours'] = eval_ours
    except Exception as e:
        print(f"警告: 无法加载 ours 方法: {e}")
    
    # PARSAC
    try:
        from parsac.line_fitter import SimplePARSACLineFitter
        from evaluate_utils import Hyperplane
        
        def eval_parsac_known(data, gt_data, gt_total_cost, num_models):
            fitter = SimplePARSACLineFitter(num_hypotheses=500, inlier_threshold=0.15)
            start = time.time()
            lines, labels = fitter.fit(data, num_models=num_models, auto_detect=False)
            runtime = time.time() - start
            
            hyperplanes = []
            for line in lines:
                hyperplanes.append(Hyperplane(normal=line[:2], distance=line[2]))
            return hyperplanes, runtime
        
        def eval_parsac_unknown(data, gt_data, gt_total_cost, num_models):
            fitter = SimplePARSACLineFitter(num_hypotheses=500, inlier_threshold=0.15)
            start = time.time()
            lines, labels = fitter.fit(data, num_models=None, auto_detect=True)
            runtime = time.time() - start
            
            hyperplanes = []
            for line in lines:
                hyperplanes.append(Hyperplane(normal=line[:2], distance=line[2]))
            return hyperplanes, runtime
        
        evaluators['parsac_known'] = eval_parsac_known
        evaluators['parsac_unknown'] = eval_parsac_unknown
    except Exception as e:
        print(f"警告: 无法加载 PARSAC 方法: {e}")
    
    # SupeRANSAC
    try:
        from superansac.sequential_ransac import SequentialRANSAC2DLine, RANSACConfig
        from evaluate_utils import Hyperplane
        
        def eval_superansac_known(data, gt_data, gt_total_cost, num_models):
            config = RANSACConfig()
            config.inlier_threshold = 0.15
            fitter = SequentialRANSAC2DLine(config)
            start = time.time()
            lines, labels = fitter.fit_known_count(data, num_models)
            runtime = time.time() - start
            
            hyperplanes = []
            for line in lines:
                hyperplanes.append(Hyperplane(normal=line[:2], distance=line[2]))
            return hyperplanes, runtime
        
        def eval_superansac_unknown(data, gt_data, gt_total_cost, num_models):
            config = RANSACConfig()
            config.inlier_threshold = 0.15
            fitter = SequentialRANSAC2DLine(config)
            start = time.time()
            lines, labels = fitter.fit(data, max_models=10)
            runtime = time.time() - start
            
            hyperplanes = []
            for line in lines:
                hyperplanes.append(Hyperplane(normal=line[:2], distance=line[2]))
            return hyperplanes, runtime
        
        evaluators['superansac_known'] = eval_superansac_known
        evaluators['superansac_unknown'] = eval_superansac_unknown
    except Exception as e:
        print(f"警告: 无法加载 SupeRANSAC 方法: {e}")
    
    # RANSAC (from compared_alg)
    try:
        from compared_alg.others.RANSAC import ransac
        from evaluate_utils import Hyperplane
        
        def eval_ransac(data, gt_data, gt_total_cost, num_models):
            alg = ransac(num_models, 2)
            alg.set_data(data)
            start = time.time()
            alg.solve()
            runtime = time.time() - start
            
            hyperplanes = []
            for k in range(alg.n):
                hyperplanes.append(Hyperplane(normal=alg.vectors[k], distance=alg.distances[k]))
            return hyperplanes, runtime
        
        evaluators['ransac'] = eval_ransac
    except Exception as e:
        print(f"警告: 无法加载 RANSAC 方法: {e}")
    
    # K-Means (from compared_alg)
    try:
        from compared_alg.others.K_Means import Kmeans
        from evaluate_utils import Hyperplane
        
        def eval_kmeans(data, gt_data, gt_total_cost, num_models):
            alg = Kmeans(num_models, 2)
            alg.set_data(data)
            start = time.time()
            alg.solve()
            runtime = time.time() - start
            
            hyperplanes = []
            for k in range(alg.n):
                hyperplanes.append(Hyperplane(normal=alg.vectors[k], distance=alg.distances[k]))
            return hyperplanes, runtime
        
        evaluators['kmeans'] = eval_kmeans
    except Exception as e:
        print(f"警告: 无法加载 K-Means 方法: {e}")
    
    # GMM (from compared_alg)
    try:
        from compared_alg.others.GMM import GMM
        from evaluate_utils import Hyperplane
        
        def eval_gmm(data, gt_data, gt_total_cost, num_models):
            alg = GMM(num_models, 2)
            alg.set_data(data)
            start = time.time()
            alg.solve()
            runtime = time.time() - start
            
            hyperplanes = []
            for k in range(alg.n):
                hyperplanes.append(Hyperplane(normal=alg.vectors[k], distance=alg.distances[k]))
            return hyperplanes, runtime
        
        evaluators['gmm'] = eval_gmm
    except Exception as e:
        print(f"警告: 无法加载 GMM 方法: {e}")
    
    return evaluators


def get_3d_evaluators():
    """获取 3D 评估函数字典"""
    evaluators = {}
    
    # Ours (流形优化方法)
    try:
        from algorithm.hyperplanes_fitting import HyperplanesFitting
        
        def eval_ours_3d(data, gt_data, gt_total_cost, num_models):
            # 使用初始值估计（自动检测超平面数量）
            alg = HyperplanesFitting(dim=3, data=data, parallel=False,
                                     method="3", whether_initial_value=True)
            start = time.time()
            hps = alg.solve(true_num=None)
            runtime = time.time() - start
            
            hyperplanes = []
            for hp in hps:
                hyperplanes.append({
                    'normal': hp.normal,
                    'distance': hp.distance
                })
            return hyperplanes, runtime
        
        evaluators['ours'] = eval_ours_3d
    except Exception as e:
        print(f"警告: 无法加载 ours 3D 方法: {e}")
    
    # PARSAC 3D
    try:
        from parsac.plane_fitter_3d import SimplePARSACPlaneFitter
        
        def eval_parsac_known_3d(data, gt_data, gt_total_cost, num_models):
            fitter = SimplePARSACPlaneFitter(num_hypotheses=300, inlier_threshold=0.2)
            start = time.time()
            planes, labels = fitter.fit(data, num_models=num_models, auto_detect=False)
            runtime = time.time() - start
            
            hyperplanes = []
            for plane in planes:
                hyperplanes.append({
                    'normal': plane[:3],
                    'distance': plane[3]
                })
            return hyperplanes, runtime
        
        def eval_parsac_unknown_3d(data, gt_data, gt_total_cost, num_models):
            fitter = SimplePARSACPlaneFitter(num_hypotheses=300, inlier_threshold=0.2)
            start = time.time()
            planes, labels = fitter.fit(data, num_models=None, auto_detect=True)
            runtime = time.time() - start
            
            hyperplanes = []
            for plane in planes:
                hyperplanes.append({
                    'normal': plane[:3],
                    'distance': plane[3]
                })
            return hyperplanes, runtime
        
        evaluators['parsac_known'] = eval_parsac_known_3d
        evaluators['parsac_unknown'] = eval_parsac_unknown_3d
    except Exception as e:
        print(f"警告: 无法加载 PARSAC 3D 方法: {e}")
    
    # SupeRANSAC 3D
    try:
        from superansac.sequential_ransac_3d import SequentialRANSAC3DPlane, RANSACConfig3D
        
        def eval_superansac_known_3d(data, gt_data, gt_total_cost, num_models):
            config = RANSACConfig3D()
            config.inlier_threshold = 0.2
            fitter = SequentialRANSAC3DPlane(config)
            start = time.time()
            planes, labels = fitter.fit_known_count(data, num_models)
            runtime = time.time() - start
            
            hyperplanes = []
            for plane in planes:
                hyperplanes.append({
                    'normal': plane[:3],
                    'distance': plane[3]
                })
            return hyperplanes, runtime
        
        def eval_superansac_unknown_3d(data, gt_data, gt_total_cost, num_models):
            config = RANSACConfig3D()
            config.inlier_threshold = 0.2
            fitter = SequentialRANSAC3DPlane(config)
            start = time.time()
            planes, labels = fitter.fit(data, max_models=10)
            runtime = time.time() - start
            
            hyperplanes = []
            for plane in planes:
                hyperplanes.append({
                    'normal': plane[:3],
                    'distance': plane[3]
                })
            return hyperplanes, runtime
        
        evaluators['superansac_known'] = eval_superansac_known_3d
        evaluators['superansac_unknown'] = eval_superansac_unknown_3d
    except Exception as e:
        print(f"警告: 无法加载 SupeRANSAC 3D 方法: {e}")
    
    # RANSAC 3D
    try:
        from compared_alg_3d.RANSAC_3D import RANSAC3D
        
        def eval_ransac_3d(data, gt_data, gt_total_cost, num_models):
            alg = RANSAC3D(num_models, 3)
            alg.set_data(data)
            start = time.time()
            alg.solve()
            runtime = time.time() - start
            
            hyperplanes = []
            for k in range(alg.n):
                hyperplanes.append({
                    'normal': alg.vectors[k],
                    'distance': alg.distances[k]
                })
            return hyperplanes, runtime
        
        evaluators['ransac'] = eval_ransac_3d
    except Exception as e:
        print(f"警告: 无法加载 RANSAC 3D 方法: {e}")
    
    # K-Means 3D
    try:
        from compared_alg_3d.K_Means_3D import KMeans3D
        
        def eval_kmeans_3d(data, gt_data, gt_total_cost, num_models):
            alg = KMeans3D(num_models, 3)
            alg.set_data(data)
            start = time.time()
            alg.solve()
            runtime = time.time() - start
            
            hyperplanes = []
            for k in range(alg.n):
                hyperplanes.append({
                    'normal': alg.vectors[k],
                    'distance': alg.distances[k]
                })
            return hyperplanes, runtime
        
        evaluators['kmeans'] = eval_kmeans_3d
    except Exception as e:
        print(f"警告: 无法加载 K-Means 3D 方法: {e}")
    
    # GMM 3D
    try:
        from compared_alg_3d.GMM_3D import GMM3D
        
        def eval_gmm_3d(data, gt_data, gt_total_cost, num_models):
            alg = GMM3D(num_models, 3)
            alg.set_data(data)
            start = time.time()
            alg.solve()
            runtime = time.time() - start
            
            hyperplanes = []
            for k in range(alg.n):
                hyperplanes.append({
                    'normal': alg.vectors[k],
                    'distance': alg.distances[k]
                })
            return hyperplanes, runtime
        
        evaluators['gmm'] = eval_gmm_3d
    except Exception as e:
        print(f"警告: 无法加载 GMM 3D 方法: {e}")
    
    return evaluators


def compute_2d_metrics(data, hyperplanes, gt_hyperplanes, gt_total_cost):
    """计算 2D 评估指标"""
    from evaluate_utils import Polyhedron, full_evaluate
    
    # 构建结果多面体
    result = Polyhedron(dim=2, hyperplanes=hyperplanes)
    
    # 构建真值多面体
    from evaluate_utils import Hyperplane
    gt_hps = []
    for j in range(len(gt_hyperplanes)):
        gt_hps.append(Hyperplane(normal=gt_hyperplanes[j][:2], distance=gt_hyperplanes[j][2]))
    ground_truth = Polyhedron(dim=2, hyperplanes=gt_hps)
    
    # 使用统一的评估函数
    eval_result = full_evaluate(data, ground_truth, gt_total_cost, result, 0)
    
    return {
        'total_cost': eval_result.total_cost,
        'cost_ratio': eval_result.cost_ratio,
        'average_distance': eval_result.average_distance,
        'total_hbar_distance': eval_result.total_hbar_distance,
        'model_count': eval_result.model_count,
        'gt_model_count': eval_result.gt_model_count,
        'model_count_error': eval_result.model_count_error
    }


def compute_3d_metrics(data, hyperplanes, gt_hyperplanes, gt_total_cost):
    """计算 3D 评估指标"""
    # 计算 Total Cost
    if len(hyperplanes) == 0:
        total_cost = float('inf')
    else:
        N = len(data)
        min_distances = np.full(N, np.inf)
        
        for hp in hyperplanes:
            distances = np.abs(np.dot(data, hp['normal']) - hp['distance'])
            min_distances = np.minimum(min_distances, distances)
        
        total_cost = np.sum(min_distances)
    
    # 计算 Cost Ratio
    cost_ratio = total_cost / gt_total_cost if gt_total_cost > 0 else float('inf')
    
    # 计算 Average Distance
    average_distance = total_cost / len(data)
    gt_average_distance = gt_total_cost / len(data)
    
    # 计算 H-bar Distance
    total_hbar_distance = 0.0
    for hp in hyperplanes:
        res_hbar = hp['distance'] * hp['normal']
        min_dist = float('inf')
        
        for gt_hp in gt_hyperplanes:
            gt_n = gt_hp[:3]
            gt_d = gt_hp[3]
            gt_hbar = gt_d * gt_n / np.linalg.norm(gt_n)
            dist = np.linalg.norm(res_hbar - gt_hbar)
            min_dist = min(min_dist, dist)
        
        total_hbar_distance += min_dist
    
    return {
        'total_cost': total_cost,
        'cost_ratio': cost_ratio,
        'average_distance': average_distance,
        'total_hbar_distance': total_hbar_distance,
        'model_count': len(hyperplanes),
        'gt_model_count': len(gt_hyperplanes),
        'model_count_error': abs(len(hyperplanes) - len(gt_hyperplanes))
    }


def run_evaluation(dim, methods, num_samples=20, output_suffix='', verbose=True):
    """
    运行评估
    
    Args:
        dim: 维度 (2 或 3)
        methods: 要评估的方法列表
        num_samples: 样本数量
        output_suffix: 输出目录后缀（如 "-2"）
        verbose: 是否打印详细信息
    """
    print(f"\n{'='*60}")
    print(f"运行 {dim}D 评估")
    print(f"{'='*60}")
    
    # 设置路径
    if dim == 2:
        data_dir = os.path.join(project_root, 'csv_dataset')
        gt_dir = os.path.join(project_root, 'csv_groundtruth')
        result_base_dir = os.path.join(project_root, 'results', f'2d{output_suffix}')
        from data.read_data import read_data_2D as read_data
        evaluators = get_2d_evaluators()
        compute_metrics = compute_2d_metrics
    else:
        data_dir = os.path.join(project_root, 'csv_dataset_3d')
        gt_dir = os.path.join(project_root, 'csv_groundtruth_3d')
        result_base_dir = os.path.join(project_root, 'results', f'3d{output_suffix}')
        from data.read_data import read_data_3D as read_data
        evaluators = get_3d_evaluators()
        compute_metrics = compute_3d_metrics
    
    # 检查数据目录
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        return
    
    # 过滤可用的方法
    available_methods = []
    for method in methods:
        if method in evaluators:
            available_methods.append(method)
        else:
            print(f"警告: 方法 {method} 不可用，跳过")
    
    if not available_methods:
        print("错误: 没有可用的方法")
        return
    
    # 结果存储
    all_results = {method: {
        'total_cost': [],
        'cost_ratio': [],
        'average_distance': [],
        'total_hbar_distance': [],
        'model_count': [],
        'gt_model_count': [],
        'model_count_error': [],
        'runtime': []
    } for method in available_methods}
    
    # 运行评估
    for i in range(num_samples):
        if verbose:
            print(f"\n处理样本 {i}...")
        
        # 读取数据
        data, gt_data, gt_total_cost = read_data(data_dir, gt_dir, file_index=i)
        num_models = len(gt_data)
        
        for method in available_methods:
            try:
                eval_func = evaluators[method]
                hyperplanes, runtime = eval_func(data, gt_data, gt_total_cost, num_models)
                
                metrics = compute_metrics(data, hyperplanes, gt_data, gt_total_cost)
                
                all_results[method]['total_cost'].append(metrics['total_cost'])
                all_results[method]['cost_ratio'].append(metrics['cost_ratio'])
                all_results[method]['average_distance'].append(metrics['average_distance'])
                all_results[method]['total_hbar_distance'].append(metrics['total_hbar_distance'])
                all_results[method]['model_count'].append(metrics['model_count'])
                all_results[method]['gt_model_count'].append(metrics['gt_model_count'])
                all_results[method]['model_count_error'].append(metrics['model_count_error'])
                all_results[method]['runtime'].append(runtime)
                
                if verbose:
                    print(f"  {method}: TC={metrics['total_cost']:.4f}, "
                          f"CR={metrics['cost_ratio']:.4f}, "
                          f"Models={metrics['model_count']}/{metrics['gt_model_count']}, "
                          f"Time={runtime:.4f}s")
                          
            except Exception as e:
                print(f"  错误 {method}: {e}")
                # 添加占位符以保持数组长度一致
                all_results[method]['total_cost'].append(float('nan'))
                all_results[method]['cost_ratio'].append(float('nan'))
                all_results[method]['average_distance'].append(float('nan'))
                all_results[method]['total_hbar_distance'].append(float('nan'))
                all_results[method]['model_count'].append(0)
                all_results[method]['gt_model_count'].append(num_models)
                all_results[method]['model_count_error'].append(num_models)
                all_results[method]['runtime'].append(float('nan'))
    
    # 保存结果
    print(f"\n{'='*60}")
    print("保存结果...")
    print(f"{'='*60}")
    
    for method in available_methods:
        method_dir = os.path.join(result_base_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        
        # 保存 CSV
        df = pd.DataFrame({
            'sample_id': list(range(num_samples)),
            'total_cost': all_results[method]['total_cost'],
            'cost_ratio': all_results[method]['cost_ratio'],
            'average_distance': all_results[method]['average_distance'],
            'total_hbar_distance': all_results[method]['total_hbar_distance'],
            'model_count': all_results[method]['model_count'],
            'gt_model_count': all_results[method]['gt_model_count'],
            'model_count_error': all_results[method]['model_count_error'],
            'runtime': all_results[method]['runtime']
        })
        csv_path = os.path.join(method_dir, f"{method}_results.csv")
        df.to_csv(csv_path, index=False)
        
        # 保存摘要
        summary_path = os.path.join(method_dir, f"{method}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"=== {method.upper()} 评估汇总 ({dim}D) ===\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"样本数: {num_samples}\n\n")
            
            tc = np.array(all_results[method]['total_cost'])
            cr = np.array(all_results[method]['cost_ratio'])
            hbar = np.array(all_results[method]['total_hbar_distance'])
            mc_err = np.array(all_results[method]['model_count_error'])
            rt = np.array(all_results[method]['runtime'])
            mc = np.array(all_results[method]['model_count'])
            
            f.write(f"平均模型数量: {np.mean(mc):.2f} ± {np.std(mc):.2f}\n")
            f.write(f"平均 Total Cost: {np.mean(tc):.4f} ± {np.std(tc):.4f}\n")
            f.write(f"平均 Cost Ratio: {np.mean(cr):.4f} ± {np.std(cr):.4f}\n")
            f.write(f"平均 Hbar Distance: {np.mean(hbar):.4f} ± {np.std(hbar):.4f}\n")
            f.write(f"平均 Model Count Error: {np.mean(mc_err):.2f} ± {np.std(mc_err):.2f}\n")
            f.write(f"平均 Runtime: {np.mean(rt):.4f}s ± {np.std(rt):.4f}s\n")
        
        print(f"已保存 {method}: {csv_path}")
    
    # 打印汇总表格
    print(f"\n{'='*60}")
    print(f"{dim}D 评估结果汇总")
    print(f"{'='*60}")
    
    print(f"\n{'Method':<20} {'TC Mean':<12} {'CR Mean':<12} {'Models':<10} {'Time':<12}")
    print("-" * 66)
    
    for method in available_methods:
        tc_mean = np.mean(all_results[method]['total_cost'])
        tc_std = np.std(all_results[method]['total_cost'])
        cr_mean = np.mean(all_results[method]['cost_ratio'])
        cr_std = np.std(all_results[method]['cost_ratio'])
        mc_mean = np.mean(all_results[method]['model_count'])
        mc_std = np.std(all_results[method]['model_count'])
        rt_mean = np.mean(all_results[method]['runtime'])
        rt_std = np.std(all_results[method]['runtime'])
        
        print(f"{method:<20} {tc_mean:.2f}±{tc_std:.2f}  {cr_mean:.2f}±{cr_std:.2f}  "
              f"{mc_mean:.1f}±{mc_std:.1f}  {rt_mean:.3f}±{rt_std:.3f}s")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='运行所有方法的评估')
    parser.add_argument('--dim', type=str, default='2', choices=['2', '3', 'all'],
                        help='评估维度: 2, 3, 或 all')
    parser.add_argument('--methods', nargs='+', 
                        default=['ours', 'parsac_known', 'parsac_unknown', 
                                'superansac_known', 'superansac_unknown',
                                'ransac', 'kmeans', 'gmm'],
                        help='要评估的方法列表')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='样本数量')
    parser.add_argument('--quiet', action='store_true',
                        help='安静模式，不打印详细信息')
    parser.add_argument('--output_suffix', type=str, default='',
                        help='输出目录后缀，例如 "-2" 会使用 results/2d-2/')
    
    args = parser.parse_args()
    
    if args.dim == '2' or args.dim == 'all':
        run_evaluation(2, args.methods, args.num_samples, 
                       output_suffix=args.output_suffix, verbose=not args.quiet)
    
    if args.dim == '3' or args.dim == 'all':
        run_evaluation(3, args.methods, args.num_samples,
                       output_suffix=args.output_suffix, verbose=not args.quiet)


if __name__ == "__main__":
    main()
