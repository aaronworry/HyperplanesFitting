"""
结果可视化脚本

提供对比不同算法结果的可视化功能。
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Dict, Optional

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts-for-eval'))


def load_results(results_dir: str) -> pd.DataFrame:
    """
    加载结果 CSV 文件
    
    Args:
        results_dir: 结果目录
    
    Returns:
        df: 结果 DataFrame
    """
    csv_path = os.path.join(results_dir, "results_summary.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def plot_fitting_result(
    points: np.ndarray,
    pred_lines: np.ndarray,
    gt_lines: np.ndarray,
    labels: np.ndarray,
    title: str = "",
    save_path: Optional[str] = None
):
    """
    绘制拟合结果
    
    Args:
        points: (N, 2) 点云
        pred_lines: (M_pred, 3) 预测直线
        gt_lines: (M_gt, 3) 真值直线
        labels: (N,) 点标签
        title: 图标题
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple']
    
    # 左图：预测结果
    ax1 = axes[0]
    for i in range(len(pred_lines)):
        mask = labels == i
        if np.any(mask):
            ax1.scatter(points[mask, 0], points[mask, 1], 
                       c=colors[i % len(colors)], alpha=0.6, s=20)
    
    # 画离群点
    mask = labels == -1
    if np.any(mask):
        ax1.scatter(points[mask, 0], points[mask, 1], c='gray', alpha=0.3, s=10)
    
    # 画预测直线
    x_range = np.array([-6, 6])
    for i, line in enumerate(pred_lines):
        n1, n2, d = line
        if abs(n2) > abs(n1):
            y_range = (d - n1 * x_range) / n2
            ax1.plot(x_range, y_range, colors[i % len(colors)] + '-', linewidth=2)
        else:
            y_range = np.array([-6, 6])
            x_plot = (d - n2 * y_range) / n1
            ax1.plot(x_plot, y_range, colors[i % len(colors)] + '-', linewidth=2)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Prediction ({len(pred_lines)} lines)')
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-6, 6)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 右图：真值
    ax2 = axes[1]
    ax2.scatter(points[:, 0], points[:, 1], c='black', alpha=0.3, s=20)
    
    # 画真值直线
    for i, line in enumerate(gt_lines):
        n1, n2, d = line
        if abs(n2) > abs(n1):
            y_range = (d - n1 * x_range) / n2
            ax2.plot(x_range, y_range, colors[i % len(colors)] + '--', linewidth=2, 
                    label=f'GT {i}')
        else:
            y_range = np.array([-6, 6])
            x_plot = (d - n2 * y_range) / n1
            ax2.plot(x_plot, y_range, colors[i % len(colors)] + '--', linewidth=2,
                    label=f'GT {i}')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'Ground Truth ({len(gt_lines)} lines)')
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-6, 6)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison_bars(
    results_dict: Dict[str, pd.DataFrame],
    metric: str,
    title: str = "",
    save_path: Optional[str] = None
):
    """
    绘制不同方法的对比柱状图
    
    Args:
        results_dict: {method_name: results_df}
        metric: 要比较的指标
        title: 图标题
        save_path: 保存路径
    """
    methods = list(results_dict.keys())
    means = []
    stds = []
    
    for method in methods:
        df = results_dict[method]
        if df is not None and metric in df.columns:
            means.append(df[metric].mean())
            stds.append(df[metric].std())
        else:
            means.append(0)
            stds.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(title if title else f'Comparison of {metric}')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上显示数值
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison_table(
    results_dict: Dict[str, pd.DataFrame],
    metrics: List[str],
    save_path: Optional[str] = None
):
    """
    生成对比表格图
    
    Args:
        results_dict: {method_name: results_df}
        metrics: 要比较的指标列表
        save_path: 保存路径
    """
    methods = list(results_dict.keys())
    
    # 构建表格数据
    table_data = []
    for method in methods:
        row = [method]
        df = results_dict[method]
        for metric in metrics:
            if df is not None and metric in df.columns:
                mean = df[metric].mean()
                std = df[metric].std()
                row.append(f"{mean:.4f} ± {std:.4f}")
            else:
                row.append("N/A")
        table_data.append(row)
    
    fig, ax = plt.subplots(figsize=(14, len(methods) * 0.6 + 1.5))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    col_labels = ['Method'] + metrics
    table = ax.table(cellText=table_data, colLabels=col_labels,
                    cellLoc='center', loc='center')
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 设置表头样式
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # 交替行背景色
    for i in range(1, len(methods) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E2F3')
    
    plt.title('Methods Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_sample(
    sample_id: int,
    data_dir: str,
    gt_dir: str,
    results_dirs: Dict[str, str],
    output_dir: str
):
    """
    可视化单个样本的所有方法结果
    
    Args:
        sample_id: 样本 ID
        data_dir: 数据目录
        gt_dir: 真值目录
        results_dirs: {method_name: results_dir}
        output_dir: 输出目录
    """
    from data_utils import read_dataset
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    points, gt_lines, _ = read_dataset(data_dir, gt_dir, sample_id)
    
    for method, results_dir in results_dirs.items():
        # 加载该方法的结果
        sample_result_path = os.path.join(results_dir, f"sample_{sample_id}.json")
        if os.path.exists(sample_result_path):
            with open(sample_result_path, 'r') as f:
                result = json.load(f)
            
            pred_lines = np.array(result['pred_lines']) if result['pred_lines'] else np.zeros((0, 3))
            labels = np.array(result['labels'])
            
            save_path = os.path.join(output_dir, f"sample_{sample_id}_{method}.png")
            plot_fitting_result(
                points, pred_lines, gt_lines, labels,
                title=f"{method} - Sample {sample_id}",
                save_path=save_path
            )
            print(f"保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='结果可视化')
    parser.add_argument('--results_base', type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'),
                       help='结果基础目录')
    parser.add_argument('--data_dir', type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv_dataset'),
                       help='数据目录')
    parser.add_argument('--gt_dir', type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv_groundtruth'),
                       help='真值目录')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'),
                       help='输出目录')
    parser.add_argument('--sample_id', type=int, default=0,
                       help='要可视化的样本 ID')
    parser.add_argument('--all_samples', action='store_true',
                       help='可视化所有样本')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找所有方法的结果
    methods = ['parsac', 'superansac']
    results_dict = {}
    results_dirs = {}
    
    for method in methods:
        method_dir = os.path.join(args.results_base, method)
        if os.path.exists(method_dir):
            df = load_results(method_dir)
            if df is not None:
                results_dict[method] = df
                results_dirs[method] = method_dir
                print(f"加载 {method}: {len(df)} 个样本")
    
    if not results_dict:
        print("未找到任何结果，请先运行评估脚本。")
        return
    
    # 1. 绘制对比表格
    metrics = ['total_cost', 'cost_ratio', 'hbar_distance', 'model_count_error', 
               'segmentation_accuracy', 'runtime']
    plot_comparison_table(
        results_dict, metrics,
        save_path=os.path.join(args.output_dir, 'comparison_table.png')
    )
    print(f"保存: {os.path.join(args.output_dir, 'comparison_table.png')}")
    
    # 2. 绘制各指标的柱状图
    for metric in metrics:
        save_path = os.path.join(args.output_dir, f'comparison_{metric}.png')
        plot_comparison_bars(
            results_dict, metric,
            title=f'Comparison of {metric}',
            save_path=save_path
        )
        print(f"保存: {save_path}")
    
    # 3. 可视化样本
    if args.all_samples:
        from data_utils import get_all_file_indices
        indices = get_all_file_indices(args.data_dir)
    else:
        indices = [args.sample_id]
    
    for sample_id in indices:
        visualize_sample(
            sample_id,
            args.data_dir,
            args.gt_dir,
            results_dirs,
            os.path.join(args.output_dir, 'samples')
        )
    
    print(f"\n所有可视化结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
