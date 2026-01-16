"""
统一评估指标模块

提供与本项目 evaluate.py 兼容的评估指标，
用于公平地比较不同算法的性能。
"""

import numpy as np
from typing import Tuple, Optional
from scipy.optimize import linear_sum_assignment


def compute_point_to_line_distances(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """
    计算每个点到每条直线的距离
    
    Args:
        points: (N, 2) 点云数据
        lines: (M, 3) 直线参数，格式为 (n1, n2, d)，n1*x + n2*y = d
    
    Returns:
        distances: (N, M) 距离矩阵
    """
    N = len(points)
    M = len(lines)
    
    distances = np.zeros((N, M))
    for j in range(M):
        n1, n2, d = lines[j]
        # 点到直线的距离（法向量已归一化）
        distances[:, j] = np.abs(n1 * points[:, 0] + n2 * points[:, 1] - d)
    
    return distances


def compute_total_cost(points: np.ndarray, lines: np.ndarray) -> float:
    """
    计算总代价函数值（所有点到最近直线的距离之和）
    
    这是本项目的核心优化目标函数。
    
    Args:
        points: (N, 2) 点云数据
        lines: (M, 3) 直线参数，格式为 (n1, n2, d)
    
    Returns:
        total_cost: 总代价
    """
    if len(lines) == 0:
        return float('inf')
    
    distances = compute_point_to_line_distances(points, lines)
    min_distances = np.min(distances, axis=1)
    total_cost = np.sum(min_distances)
    
    return total_cost


def compute_average_distance(points: np.ndarray, lines: np.ndarray) -> float:
    """
    计算平均距离（每个点到最近直线的平均距离）
    
    Args:
        points: (N, 2) 点云数据
        lines: (M, 3) 直线参数，格式为 (n1, n2, d)
    
    Returns:
        average_distance: 平均距离
    """
    total_cost = compute_total_cost(points, lines)
    return total_cost / len(points)


def compute_hbar(lines: np.ndarray) -> np.ndarray:
    """
    计算 hbar 向量（用于直线匹配）
    
    hbar = d * n，其中 n 是法向量，d 是到原点的距离
    
    Args:
        lines: (M, 3) 直线参数，格式为 (n1, n2, d)
    
    Returns:
        hbar: (M, 2) 每条直线的 hbar 向量
    """
    M = len(lines)
    hbar = np.zeros((M, 2))
    for i in range(M):
        n1, n2, d = lines[i]
        hbar[i, :] = d * np.array([n1, n2])
    return hbar


def compute_hbar_distance(pred_lines: np.ndarray, gt_lines: np.ndarray) -> float:
    """
    计算预测直线与真值直线之间的 hbar 距离
    
    使用贪心匹配：每条预测直线匹配到最近的真值直线
    
    Args:
        pred_lines: (M_pred, 3) 预测直线参数
        gt_lines: (M_gt, 3) 真值直线参数
    
    Returns:
        total_hbar_distance: 总 hbar 距离
    """
    if len(pred_lines) == 0:
        return float('inf')
    
    hbar_pred = compute_hbar(pred_lines)
    hbar_gt = compute_hbar(gt_lines)
    
    # 计算距离矩阵
    M_pred = len(pred_lines)
    M_gt = len(gt_lines)
    
    dist_matrix = np.zeros((M_pred, M_gt))
    for i in range(M_pred):
        for j in range(M_gt):
            dist_matrix[i, j] = np.linalg.norm(hbar_pred[i] - hbar_gt[j])
    
    # 贪心匹配：每条预测直线匹配到最近的真值直线
    total_distance = np.sum(np.min(dist_matrix, axis=1))
    
    return total_distance


def compute_hbar_distance_optimal(pred_lines: np.ndarray, gt_lines: np.ndarray) -> float:
    """
    使用匈牙利算法计算最优匹配的 hbar 距离
    
    Args:
        pred_lines: (M_pred, 3) 预测直线参数
        gt_lines: (M_gt, 3) 真值直线参数
    
    Returns:
        total_hbar_distance: 总 hbar 距离
    """
    if len(pred_lines) == 0 or len(gt_lines) == 0:
        return float('inf')
    
    hbar_pred = compute_hbar(pred_lines)
    hbar_gt = compute_hbar(gt_lines)
    
    M_pred = len(pred_lines)
    M_gt = len(gt_lines)
    
    # 计算距离矩阵
    dist_matrix = np.zeros((M_pred, M_gt))
    for i in range(M_pred):
        for j in range(M_gt):
            dist_matrix[i, j] = np.linalg.norm(hbar_pred[i] - hbar_gt[j])
    
    # 使用匈牙利算法找最优匹配
    # 如果预测数量与真值数量不同，需要处理
    if M_pred <= M_gt:
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        total_distance = dist_matrix[row_ind, col_ind].sum()
    else:
        # 预测多于真值，对列进行匹配
        col_ind, row_ind = linear_sum_assignment(dist_matrix.T)
        total_distance = dist_matrix[row_ind, col_ind].sum()
        # 加上未匹配预测直线的惩罚
        # 这里使用它们到最近真值的距离
        all_matched = set(row_ind)
        for i in range(M_pred):
            if i not in all_matched:
                total_distance += np.min(dist_matrix[i, :])
    
    return total_distance


def compute_model_count_error(pred_lines: np.ndarray, gt_lines: np.ndarray) -> int:
    """
    计算模型数量误差
    
    Args:
        pred_lines: 预测直线
        gt_lines: 真值直线
    
    Returns:
        error: 数量差异的绝对值
    """
    return abs(len(pred_lines) - len(gt_lines))


def compute_segmentation_accuracy(
    points: np.ndarray, 
    pred_lines: np.ndarray, 
    gt_lines: np.ndarray,
    gt_labels: Optional[np.ndarray] = None
) -> float:
    """
    计算分割准确率
    
    将每个点分配到最近的预测直线，然后与真值标签比较
    
    Args:
        points: (N, 2) 点云数据
        pred_lines: (M_pred, 3) 预测直线
        gt_lines: (M_gt, 3) 真值直线
        gt_labels: (N,) 真值标签（可选，如果没有则自动计算）
    
    Returns:
        accuracy: 分割准确率 (0 到 1)
    """
    N = len(points)
    
    if gt_labels is None:
        # 根据真值直线计算真值标签
        gt_distances = compute_point_to_line_distances(points, gt_lines)
        gt_labels = np.argmin(gt_distances, axis=1)
    
    if len(pred_lines) == 0:
        return 0.0
    
    # 计算预测标签
    pred_distances = compute_point_to_line_distances(points, pred_lines)
    pred_labels = np.argmin(pred_distances, axis=1)
    
    # 找到预测直线与真值直线的最优对应关系
    M_pred = len(pred_lines)
    M_gt = len(gt_lines)
    
    hbar_pred = compute_hbar(pred_lines)
    hbar_gt = compute_hbar(gt_lines)
    
    # 计算直线之间的距离矩阵
    line_dist_matrix = np.zeros((M_pred, M_gt))
    for i in range(M_pred):
        for j in range(M_gt):
            line_dist_matrix[i, j] = np.linalg.norm(hbar_pred[i] - hbar_gt[j])
    
    # 匈牙利算法找最优匹配
    if M_pred <= M_gt:
        row_ind, col_ind = linear_sum_assignment(line_dist_matrix)
        pred_to_gt = dict(zip(row_ind, col_ind))
    else:
        col_ind, row_ind = linear_sum_assignment(line_dist_matrix.T)
        pred_to_gt = dict(zip(row_ind, col_ind))
    
    # 将预测标签映射到真值标签空间
    mapped_pred_labels = np.zeros(N, dtype=int)
    for i in range(N):
        pred_label = pred_labels[i]
        if pred_label in pred_to_gt:
            mapped_pred_labels[i] = pred_to_gt[pred_label]
        else:
            # 未匹配的预测直线，找最近的真值直线
            mapped_pred_labels[i] = np.argmin(line_dist_matrix[pred_label, :])
    
    # 计算准确率
    accuracy = np.mean(mapped_pred_labels == gt_labels)
    
    return accuracy


def evaluate_result(
    points: np.ndarray,
    pred_lines: np.ndarray,
    gt_lines: np.ndarray,
    gt_total_cost: float,
    gt_labels: Optional[np.ndarray] = None
) -> dict:
    """
    综合评估结果
    
    Args:
        points: (N, 2) 点云数据
        pred_lines: (M_pred, 3) 预测直线参数
        gt_lines: (M_gt, 3) 真值直线参数
        gt_total_cost: 真值总代价
        gt_labels: (N,) 真值标签（可选）
    
    Returns:
        metrics: 评估指标字典
    """
    N = len(points)
    M_pred = len(pred_lines)
    M_gt = len(gt_lines)
    
    # 计算各项指标
    total_cost = compute_total_cost(points, pred_lines)
    average_distance = total_cost / N
    gt_average_distance = gt_total_cost / N
    hbar_distance = compute_hbar_distance(pred_lines, gt_lines)
    hbar_distance_optimal = compute_hbar_distance_optimal(pred_lines, gt_lines)
    model_count_error = compute_model_count_error(pred_lines, gt_lines)
    
    # 分割准确率
    seg_accuracy = compute_segmentation_accuracy(points, pred_lines, gt_lines, gt_labels)
    
    metrics = {
        'total_cost': total_cost,
        'average_distance': average_distance,
        'gt_total_cost': gt_total_cost,
        'gt_average_distance': gt_average_distance,
        'cost_ratio': total_cost / gt_total_cost if gt_total_cost > 0 else float('inf'),
        'hbar_distance': hbar_distance,
        'hbar_distance_optimal': hbar_distance_optimal,
        'model_count_pred': M_pred,
        'model_count_gt': M_gt,
        'model_count_error': model_count_error,
        'segmentation_accuracy': seg_accuracy,
        'num_points': N
    }
    
    return metrics


def print_metrics(metrics: dict, method_name: str = ""):
    """
    打印评估指标
    
    Args:
        metrics: 评估指标字典
        method_name: 方法名称
    """
    if method_name:
        print(f"\n=== {method_name} ===")
    
    print(f"  Total Cost:          {metrics['total_cost']:.4f}")
    print(f"  GT Total Cost:       {metrics['gt_total_cost']:.4f}")
    print(f"  Cost Ratio:          {metrics['cost_ratio']:.4f}")
    print(f"  Average Distance:    {metrics['average_distance']:.6f}")
    print(f"  GT Average Distance: {metrics['gt_average_distance']:.6f}")
    print(f"  Hbar Distance:       {metrics['hbar_distance']:.4f}")
    print(f"  Model Count:         {metrics['model_count_pred']} (GT: {metrics['model_count_gt']})")
    print(f"  Model Count Error:   {metrics['model_count_error']}")
    print(f"  Seg. Accuracy:       {metrics['segmentation_accuracy']:.4f}")


if __name__ == "__main__":
    # 测试代码
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from data_utils import read_dataset, get_all_file_indices
    
    # 路径设置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "csv_dataset")
    gt_dir = os.path.join(base_dir, "csv_groundtruth")
    
    # 读取数据
    indices = get_all_file_indices(data_dir)
    if indices:
        points, gt_lines, gt_total_cost = read_dataset(data_dir, gt_dir, indices[0])
        
        # 使用真值作为预测来测试
        print("测试：使用真值作为预测")
        metrics = evaluate_result(points, gt_lines, gt_lines, gt_total_cost)
        print_metrics(metrics, "Ground Truth")
        
        # 添加噪声测试
        print("\n测试：添加噪声到真值")
        noisy_lines = gt_lines.copy()
        noisy_lines[:, 2] += np.random.randn(len(gt_lines)) * 0.1  # 添加 d 的噪声
        metrics = evaluate_result(points, noisy_lines, gt_lines, gt_total_cost)
        print_metrics(metrics, "Noisy Prediction")
