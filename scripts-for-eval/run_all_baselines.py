"""
批量运行所有基线方法

该脚本统一运行所有评估方法并汇总结果。
"""

import os
import sys
import time
import json
import argparse
import subprocess
import pandas as pd
from typing import Dict, List

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_superansac(
    data_dir: str,
    gt_dir: str,
    output_dir: str,
    inlier_threshold: float = 0.1,
    max_iterations: int = 1000,
    max_models: int = 10,
    known_count: bool = False,
    scoring: str = 'msac'
) -> pd.DataFrame:
    """运行 Sequential RANSAC 评估"""
    from superansac.evaluate_superansac import evaluate_on_dataset
    from superansac.sequential_ransac import RANSACConfig
    
    config = RANSACConfig()
    config.inlier_threshold = inlier_threshold
    config.max_iterations = max_iterations
    config.scoring = scoring
    
    return evaluate_on_dataset(
        data_dir, gt_dir, output_dir, config,
        max_models=max_models,
        known_model_count=known_count,
        verbose=True
    )


def run_parsac(
    data_dir: str,
    gt_dir: str,
    output_dir: str,
    num_hypotheses: int = 200,
    num_instances: int = 4,
    inlier_threshold: float = 0.1,
    inlier_softness: float = 5.0,
    known_count: bool = False
) -> pd.DataFrame:
    """运行 PARSAC 评估"""
    from parsac.evaluate_parsac import evaluate_on_dataset
    
    return evaluate_on_dataset(
        data_dir, gt_dir, output_dir,
        num_hypotheses=num_hypotheses,
        num_instances=num_instances,
        inlier_threshold=inlier_threshold,
        inlier_softness=inlier_softness,
        known_model_count=known_count,
        verbose=True
    )


def run_sklearn_ransac(
    data_dir: str,
    gt_dir: str,
    output_dir: str,
    residual_threshold: float = 0.1,
    max_models: int = 10
) -> pd.DataFrame:
    """
    运行 sklearn RANSAC（Sequential RANSAC 策略）
    
    使用 sklearn 的 RANSACRegressor 进行多直线拟合
    """
    import numpy as np
    from sklearn.linear_model import RANSACRegressor
    from data_utils import read_dataset, get_all_file_indices
    from metrics import evaluate_result, print_metrics
    
    os.makedirs(output_dir, exist_ok=True)
    indices = get_all_file_indices(data_dir)
    all_results = []
    
    for idx in indices:
        print(f"\n处理样本 {idx}...")
        points, gt_lines, gt_total_cost = read_dataset(data_dir, gt_dir, idx)
        
        start_time = time.time()
        
        # Sequential RANSAC with sklearn
        remaining_mask = np.ones(len(points), dtype=bool)
        labels = -np.ones(len(points), dtype=int)
        lines = []
        
        for model_idx in range(max_models):
            remaining_points = points[remaining_mask]
            remaining_indices = np.where(remaining_mask)[0]
            
            if len(remaining_points) < 5:
                break
            
            try:
                # sklearn RANSAC
                ransac = RANSACRegressor(
                    residual_threshold=residual_threshold,
                    max_trials=1000,
                    random_state=42 + model_idx
                )
                X = remaining_points[:, 0].reshape(-1, 1)
                y = remaining_points[:, 1]
                
                ransac.fit(X, y)
                
                # 获取内点
                inlier_mask_local = ransac.inlier_mask_
                
                if np.sum(inlier_mask_local) < 5:
                    break
                
                # 计算直线参数
                # y = ax + b  =>  ax - y + b = 0
                a = ransac.estimator_.coef_[0]
                b = -1
                c = ransac.estimator_.intercept_
                
                # 归一化
                norm = np.sqrt(a**2 + b**2)
                n1, n2 = a / norm, b / norm
                d = -c / norm
                
                lines.append([n1, n2, d])
                
                # 更新标签和剩余点
                inlier_indices = remaining_indices[inlier_mask_local]
                labels[inlier_indices] = model_idx
                remaining_mask[inlier_indices] = False
                
            except Exception as e:
                print(f"  警告: RANSAC 拟合失败 - {e}")
                break
        
        runtime = time.time() - start_time
        
        pred_lines = np.array(lines) if lines else np.zeros((0, 3))
        
        metrics = evaluate_result(points, pred_lines, gt_lines, gt_total_cost)
        metrics['sample_id'] = idx
        metrics['runtime'] = runtime
        
        print_metrics(metrics, f"Sample {idx}")
        print(f"  Runtime: {runtime:.4f}s")
        
        all_results.append(metrics)
        
        # 保存单个样本结果
        sample_result = {
            'pred_lines': pred_lines.tolist() if len(pred_lines) > 0 else [],
            'labels': labels.tolist(),
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                       for k, v in metrics.items()}
        }
        with open(os.path.join(output_dir, f"sample_{idx}.json"), 'w') as f:
            json.dump(sample_result, f, indent=2)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, "results_summary.csv"), index=False)
    
    return results_df


def combine_results(results_base: str, methods: List[str]) -> pd.DataFrame:
    """
    合并所有方法的结果
    
    Args:
        results_base: 结果基础目录
        methods: 方法列表
    
    Returns:
        combined_df: 合并的 DataFrame
    """
    all_summaries = []
    
    for method in methods:
        method_dir = os.path.join(results_base, method)
        csv_path = os.path.join(method_dir, "results_summary.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['method'] = method
            all_summaries.append(df)
    
    if all_summaries:
        combined_df = pd.concat(all_summaries, ignore_index=True)
        return combined_df
    return None


def print_summary_table(combined_df: pd.DataFrame, metrics: List[str]):
    """打印汇总表格"""
    print("\n" + "=" * 80)
    print("方法对比汇总")
    print("=" * 80)
    
    # 按方法分组计算统计量
    summary = combined_df.groupby('method').agg({
        m: ['mean', 'std'] for m in metrics if m in combined_df.columns
    }).round(4)
    
    print(summary.to_string())


def main():
    parser = argparse.ArgumentParser(description='批量运行所有基线方法')
    parser.add_argument('--data_dir', type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csv_dataset'),
                       help='数据目录')
    parser.add_argument('--gt_dir', type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csv_groundtruth'),
                       help='真值目录')
    parser.add_argument('--results_base', type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results'),
                       help='结果基础目录')
    parser.add_argument('--inlier_threshold', type=float, default=0.1,
                       help='内点阈值')
    parser.add_argument('--known_count', action='store_true',
                       help='是否使用已知的模型数量')
    parser.add_argument('--methods', nargs='+', 
                       default=['superansac', 'parsac', 'sklearn_ransac'],
                       help='要运行的方法')
    parser.add_argument('--skip_existing', action='store_true',
                       help='跳过已有结果的方法')
    
    args = parser.parse_args()
    
    os.makedirs(args.results_base, exist_ok=True)
    
    print("=" * 80)
    print("批量运行基线方法")
    print("=" * 80)
    print(f"数据目录: {args.data_dir}")
    print(f"真值目录: {args.gt_dir}")
    print(f"结果目录: {args.results_base}")
    print(f"要运行的方法: {args.methods}")
    print(f"内点阈值: {args.inlier_threshold}")
    print(f"已知模型数: {args.known_count}")
    
    results = {}
    
    # 运行各方法
    for method in args.methods:
        output_dir = os.path.join(args.results_base, method)
        
        # 检查是否跳过
        if args.skip_existing and os.path.exists(os.path.join(output_dir, "results_summary.csv")):
            print(f"\n跳过 {method} (结果已存在)")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"运行 {method}")
        print("=" * 60)
        
        try:
            if method == 'superansac':
                results[method] = run_superansac(
                    args.data_dir, args.gt_dir, output_dir,
                    inlier_threshold=args.inlier_threshold,
                    known_count=args.known_count
                )
            elif method == 'parsac':
                results[method] = run_parsac(
                    args.data_dir, args.gt_dir, output_dir,
                    inlier_threshold=args.inlier_threshold,
                    known_count=args.known_count
                )
            elif method == 'sklearn_ransac':
                results[method] = run_sklearn_ransac(
                    args.data_dir, args.gt_dir, output_dir,
                    residual_threshold=args.inlier_threshold
                )
            else:
                print(f"未知方法: {method}")
                continue
            
            print(f"\n{method} 完成，结果保存到: {output_dir}")
            
        except Exception as e:
            print(f"\n{method} 运行失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 合并结果
    print("\n" + "=" * 80)
    print("合并所有结果")
    print("=" * 80)
    
    combined_df = combine_results(args.results_base, args.methods)
    
    if combined_df is not None:
        # 保存合并结果
        combined_path = os.path.join(args.results_base, "combined_results.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"合并结果保存到: {combined_path}")
        
        # 打印汇总
        metrics = ['total_cost', 'cost_ratio', 'hbar_distance', 
                  'model_count_error', 'segmentation_accuracy', 'runtime']
        print_summary_table(combined_df, metrics)
    else:
        print("没有找到任何结果")
    
    print("\n完成！")


if __name__ == "__main__":
    main()
