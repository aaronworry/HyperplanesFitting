#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估工具模块 - 与项目中的 evaluate.py 完全一致的评估指标

此模块实现了论文中定义的所有评估指标：
- total_cost: 所有点到其最近拟合超平面的距离总和
- average_distance: 平均每点的拟合误差
- total_hbar_distance: 拟合超平面与真值超平面的 h-bar 距离总和
- ground_truth_average_distance: 真值的平均距离

注意：此评估使用与论文完全相同的 h-bar 定义和距离计算方式
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Hyperplane:
    """超平面类，与 algorithm/initial_value.py 中的 Hyperplane 类接口一致"""
    n: np.ndarray   # 单位法向量
    d: float        # 有符号距离 (n^T x = d)
    
    def __init__(self, normal: np.ndarray, distance: float):
        self.n = np.array(normal)
        self.d = distance


@dataclass
class Polyhedron:
    """多面体类（超平面集合），与 algorithm/initial_value.py 中的 Polyhedron 类接口一致"""
    dim: int
    hyperplanes: List[Hyperplane]
    
    def __init__(self, dim: int, hyperplanes: List[Hyperplane]):
        self.dim = dim
        self.hyperplanes = hyperplanes
    
    @property
    def A(self) -> np.ndarray:
        """返回法向量矩阵 (m x d)"""
        return np.array([hp.n for hp in self.hyperplanes])
    
    @property
    def b(self) -> np.ndarray:
        """返回偏置向量 (-d)，即 Ax + b = 0 形式的 b"""
        return np.array([-hp.d for hp in self.hyperplanes])


def cal_hbar(dim: int, polygon: Polyhedron) -> np.ndarray:
    """
    计算超平面的 h-bar 表示
    
    论文中的 h-bar 定义：对于超平面 n^T x = d，其 h-bar = d * n
    这是一种将法向量和距离统一编码的表示方法
    
    Args:
        dim: 空间维度
        polygon: 超平面集合
        
    Returns:
        hbar: (num_hyperplanes, dim) 的 h-bar 矩阵
    """
    A = polygon.A  # 法向量矩阵
    b = polygon.b  # -d 向量
    num = len(A)
    hbar = np.zeros((num, dim))
    for i in range(num):
        # hbar[i] = -b[i] * A[i] = d[i] * n[i]
        hbar[i, :] = -1.0 * b[i] * A[i, :]
    return hbar


def evaluate(data: np.ndarray, 
             ground_truth: Polyhedron, 
             ground_truth_total_cost: float, 
             result: Polyhedron) -> Tuple[float, float, float, float]:
    """
    评估拟合结果，与项目中的 evaluate.py 完全一致
    
    Args:
        data: 点云数据 (n x d)
        ground_truth: 真值超平面集合
        ground_truth_total_cost: 真值的总代价 (sum of min distances to ground truth)
        result: 拟合结果超平面集合
        
    Returns:
        total_hbar_distance: h-bar 距离总和
        total_cost: 总代价
        average_distance: 平均距离
        ground_truth_average_distance: 真值平均距离
    """
    N_matrix = result.A      # 拟合结果的法向量矩阵
    d_matrix = -1 * result.b  # 拟合结果的距离向量
    
    # compute total_cost: sum of min distances to fitted hyperplanes
    n, m = len(data), len(N_matrix)
    dim = len(data[0])
    total_cost = 0.0
    
    for i in range(n):
        # 点 x_i 到所有超平面的距离：|n_j^T x_i - d_j|
        temp = np.abs(N_matrix @ data[i, :] - d_matrix)
        total_cost += np.min(temp)
    
    # average_distance
    average_distance = total_cost / n
    ground_truth_average_distance = ground_truth_total_cost / float(n)
    
    # 计算 h-bar 距离
    M_result = cal_hbar(dim, result)
    M_ground_truth = cal_hbar(dim, ground_truth)
    
    len_G = len(ground_truth.A)
    len_R = len(N_matrix)
    
    # 计算 h-bar 距离矩阵
    M_hbar_distance = np.zeros((len_R, len_G))
    
    for i in range(len_R):
        for j in range(len_G):
            M_hbar_distance[i][j] = np.linalg.norm(M_result[i, :] - M_ground_truth[j, :])
    
    # 总 h-bar 距离：每个拟合超平面到其最近真值超平面的 h-bar 距离之和
    total_hbar_distance = np.sum(np.min(M_hbar_distance, axis=1))
    
    return total_hbar_distance, total_cost, average_distance, ground_truth_average_distance


def compute_model_count_error(fitted_count: int, gt_count: int) -> int:
    """计算模型数量误差"""
    return abs(fitted_count - gt_count)


def compute_cost_ratio(total_cost: float, gt_total_cost: float) -> float:
    """计算代价比率"""
    if gt_total_cost > 0:
        return total_cost / gt_total_cost
    return float('inf')


@dataclass
class EvaluationResult:
    """完整的评估结果"""
    total_hbar_distance: float
    total_cost: float
    average_distance: float
    ground_truth_average_distance: float
    cost_ratio: float
    model_count: int
    gt_model_count: int
    model_count_error: int
    runtime: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'total_hbar_distance': self.total_hbar_distance,
            'total_cost': self.total_cost,
            'average_distance': self.average_distance,
            'ground_truth_average_distance': self.ground_truth_average_distance,
            'cost_ratio': self.cost_ratio,
            'model_count': self.model_count,
            'gt_model_count': self.gt_model_count,
            'model_count_error': self.model_count_error,
            'runtime': self.runtime
        }


def full_evaluate(data: np.ndarray,
                  ground_truth: Polyhedron,
                  ground_truth_total_cost: float,
                  result: Polyhedron,
                  runtime: float = 0.0) -> EvaluationResult:
    """
    完整评估，返回所有指标
    
    Args:
        data: 点云数据
        ground_truth: 真值超平面集合
        ground_truth_total_cost: 真值总代价
        result: 拟合结果
        runtime: 运行时间
        
    Returns:
        EvaluationResult: 包含所有评估指标的结果对象
    """
    total_hbar_distance, total_cost, average_distance, gt_avg_distance = evaluate(
        data, ground_truth, ground_truth_total_cost, result
    )
    
    model_count = len(result.hyperplanes)
    gt_model_count = len(ground_truth.hyperplanes)
    
    return EvaluationResult(
        total_hbar_distance=total_hbar_distance,
        total_cost=total_cost,
        average_distance=average_distance,
        ground_truth_average_distance=gt_avg_distance,
        cost_ratio=compute_cost_ratio(total_cost, ground_truth_total_cost),
        model_count=model_count,
        gt_model_count=gt_model_count,
        model_count_error=compute_model_count_error(model_count, gt_model_count),
        runtime=runtime
    )
