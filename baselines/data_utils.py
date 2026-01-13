"""
数据读取和格式转换工具

用于读取 csv_dataset 和 csv_groundtruth 中的数据，
并提供统一的数据格式供各种算法使用。
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple, List, Optional


def read_csv_data(data_path: str) -> np.ndarray:
    """
    读取点云数据 CSV 文件
    
    Args:
        data_path: CSV 文件路径
    
    Returns:
        points: (N, 2) numpy 数组，包含所有点的 (x, y) 坐标
    """
    df = pd.read_csv(data_path)
    points = df[['x', 'y']].values
    return points


def read_csv_groundtruth(gt_path: str) -> Tuple[np.ndarray, float]:
    """
    读取真值数据 CSV 文件
    
    Args:
        gt_path: 真值 CSV 文件路径
    
    Returns:
        lines: (M, 3) numpy 数组，每行为 (n1, n2, d)，表示直线 n1*x + n2*y = d
        total_distance: 所有点到真值直线的总距离
    """
    df = pd.read_csv(gt_path)
    lines = df[['one', 'two', 'd']].values
    total_distance = df['totaldistance'].iloc[0]
    return lines, total_distance


def read_dataset(data_dir: str, gt_dir: str, file_index: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    读取一个完整的数据样本
    
    Args:
        data_dir: 数据目录路径
        gt_dir: 真值目录路径
        file_index: 文件索引
    
    Returns:
        points: (N, 2) 点云数据
        gt_lines: (M, 3) 真值直线参数
        total_distance: 真值总距离
    """
    data_path = os.path.join(data_dir, f"{file_index}.csv")
    gt_path = os.path.join(gt_dir, f"{file_index}.csv")
    
    points = read_csv_data(data_path)
    gt_lines, total_distance = read_csv_groundtruth(gt_path)
    
    return points, gt_lines, total_distance


def get_dataset_size(data_dir: str) -> int:
    """
    获取数据集中的样本数量
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        count: 样本数量
    """
    count = 0
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            count += 1
    return count


def get_all_file_indices(data_dir: str) -> List[int]:
    """
    获取数据集中所有文件的索引
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        indices: 文件索引列表
    """
    indices = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            try:
                idx = int(filename.replace('.csv', ''))
                indices.append(idx)
            except ValueError:
                continue
    return sorted(indices)


def assign_points_to_lines(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """
    将点分配到最近的直线
    
    Args:
        points: (N, 2) 点云数据
        lines: (M, 3) 直线参数，每行为 (n1, n2, d)
    
    Returns:
        labels: (N,) 每个点的直线标签 (0 到 M-1)
    """
    N = len(points)
    M = len(lines)
    
    # 计算每个点到每条直线的距离
    distances = np.zeros((N, M))
    for j in range(M):
        n1, n2, d = lines[j]
        # 点到直线 n1*x + n2*y = d 的距离
        distances[:, j] = np.abs(n1 * points[:, 0] + n2 * points[:, 1] - d)
    
    # 分配到最近的直线
    labels = np.argmin(distances, axis=1)
    return labels


def line_params_to_ax_b_c(lines: np.ndarray) -> np.ndarray:
    """
    将直线参数从 (n1, n2, d) 格式转换为 ax + by + c = 0 格式
    
    Args:
        lines: (M, 3) 直线参数，格式为 n1*x + n2*y = d
    
    Returns:
        lines_abc: (M, 3) 直线参数，格式为 ax + by + c = 0
    """
    lines_abc = np.zeros_like(lines)
    lines_abc[:, 0] = lines[:, 0]  # a = n1
    lines_abc[:, 1] = lines[:, 1]  # b = n2
    lines_abc[:, 2] = -lines[:, 2]  # c = -d
    return lines_abc


def ax_b_c_to_line_params(lines_abc: np.ndarray) -> np.ndarray:
    """
    将直线参数从 ax + by + c = 0 格式转换为 (n1, n2, d) 格式
    并归一化法向量
    
    Args:
        lines_abc: (M, 3) 直线参数，格式为 ax + by + c = 0
    
    Returns:
        lines: (M, 3) 直线参数，格式为 n1*x + n2*y = d，法向量已归一化
    """
    lines = np.zeros_like(lines_abc)
    for i in range(len(lines_abc)):
        a, b, c = lines_abc[i]
        norm = np.sqrt(a**2 + b**2)
        if norm > 1e-10:
            lines[i, 0] = a / norm  # n1
            lines[i, 1] = b / norm  # n2
            lines[i, 2] = -c / norm  # d
    return lines


def fit_line_from_points(points: np.ndarray) -> np.ndarray:
    """
    使用 SVD 从点集拟合直线
    
    Args:
        points: (N, 2) 点云数据
    
    Returns:
        line: (3,) 直线参数 (n1, n2, d)，法向量已归一化
    """
    if len(points) < 2:
        return np.array([0, 0, 0])
    
    # 中心化
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # SVD
    U, S, Vt = np.linalg.svd(centered)
    
    # 法向量是最小奇异值对应的右奇异向量
    normal = Vt[-1]
    
    # 归一化
    norm = np.linalg.norm(normal)
    if norm > 1e-10:
        normal = normal / norm
    
    # d = n · centroid
    d = np.dot(normal, centroid)
    
    return np.array([normal[0], normal[1], d])


class LineResult:
    """
    存储直线拟合结果的类，与原项目格式兼容
    """
    def __init__(self, lines: np.ndarray):
        """
        Args:
            lines: (M, 3) 直线参数，格式为 n1*x + n2*y = d
        """
        # A 矩阵存储法向量 (M, 2)
        self.A = lines[:, :2]
        # b 向量存储 -d (M,)
        self.b = -lines[:, 2]
    
    @classmethod
    def from_A_b(cls, A: np.ndarray, b: np.ndarray):
        """
        从 A 和 b 创建 LineResult
        
        Args:
            A: (M, 2) 法向量矩阵
            b: (M,) -d 向量
        """
        M = len(A)
        lines = np.zeros((M, 3))
        lines[:, :2] = A
        lines[:, 2] = -b
        return cls(lines)
    
    def get_lines(self) -> np.ndarray:
        """
        获取直线参数
        
        Returns:
            lines: (M, 3) 直线参数，格式为 (n1, n2, d)
        """
        M = len(self.A)
        lines = np.zeros((M, 3))
        lines[:, :2] = self.A
        lines[:, 2] = -self.b
        return lines


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csv_dataset")
    gt_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csv_groundtruth")
    
    print(f"数据目录: {data_dir}")
    print(f"真值目录: {gt_dir}")
    
    # 测试读取
    indices = get_all_file_indices(data_dir)
    print(f"数据集大小: {len(indices)}")
    print(f"文件索引: {indices}")
    
    # 读取第一个样本
    if indices:
        points, gt_lines, total_distance = read_dataset(data_dir, gt_dir, indices[0])
        print(f"\n样本 {indices[0]}:")
        print(f"  点数: {len(points)}")
        print(f"  直线数: {len(gt_lines)}")
        print(f"  真值总距离: {total_distance:.4f}")
        print(f"  直线参数:\n{gt_lines}")
        
        # 测试点分配
        labels = assign_points_to_lines(points, gt_lines)
        print(f"  点分配: {np.bincount(labels)}")
