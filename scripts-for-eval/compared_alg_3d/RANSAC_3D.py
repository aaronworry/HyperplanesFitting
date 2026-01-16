#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D RANSAC 平面拟合算法

使用 Sequential RANSAC 方法在 3D 点云中拟合多个平面。

作者: Hyperplanes Fitting Team
"""

import numpy as np
from typing import List, Tuple, Optional


class RANSAC3D:
    """
    3D RANSAC 平面拟合器
    
    使用 Sequential RANSAC 方法从 3D 点云中拟合多个平面。
    """
    
    def __init__(self, n: int, dim: int = 3):
        """
        初始化 RANSAC 3D 拟合器
        
        Args:
            n: 要拟合的平面数量
            dim: 数据维度 (必须为 3)
        """
        self.n = n
        self.dim = dim
        self.vectors = np.zeros((n, dim))
        self.distances = np.zeros(n)
        
        self.data = None
        self.number = 0
    
    def set_data(self, data: np.ndarray):
        """
        设置数据
        
        Args:
            data: 3D 点云数据 (N x 3)
        """
        self.data = data
        self.number = len(data)
    
    def solve(self):
        """求解平面拟合"""
        planes = self._fit_planes()
        self.n = len(planes)
        self.vectors = np.zeros((self.n, self.dim))
        self.distances = np.zeros(self.n)
        
        for i, (normal, d) in enumerate(planes):
            self.vectors[i] = normal
            self.distances[i] = d
    
    def _fit_planes(self, 
                    max_iterations: int = 1000,
                    inlier_threshold: float = 0.3,
                    min_inliers: int = 15) -> List[Tuple[np.ndarray, float]]:
        """
        使用 Sequential RANSAC 拟合多个平面
        
        Args:
            max_iterations: RANSAC 最大迭代次数
            inlier_threshold: 内点阈值
            min_inliers: 最小内点数
            
        Returns:
            平面列表 [(normal, d), ...]
        """
        remaining_points = self.data.copy()
        planes = []
        
        while len(remaining_points) >= min_inliers and len(planes) < self.n:
            best_plane = None
            best_inliers = None
            best_num_inliers = 0
            
            for _ in range(max_iterations):
                # 随机采样 3 个点
                if len(remaining_points) < 3:
                    break
                    
                idx = np.random.choice(len(remaining_points), 3, replace=False)
                p1, p2, p3 = remaining_points[idx]
                
                # 计算平面法向量
                v1 = p2 - p1
                v2 = p3 - p1
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                
                if norm < 1e-10:  # 三点共线
                    continue
                
                normal = normal / norm
                d = np.dot(normal, p1)
                
                # 计算所有点到平面的距离
                distances = np.abs(np.dot(remaining_points, normal) - d)
                
                # 统计内点
                inlier_mask = distances < inlier_threshold
                num_inliers = np.sum(inlier_mask)
                
                if num_inliers > best_num_inliers:
                    best_num_inliers = num_inliers
                    best_plane = (normal, d)
                    best_inliers = inlier_mask
            
            if best_plane is None or best_num_inliers < min_inliers:
                break
            
            # 使用所有内点重新拟合平面 (SVD)
            inlier_points = remaining_points[best_inliers]
            centroid = np.mean(inlier_points, axis=0)
            centered = inlier_points - centroid
            _, _, Vt = np.linalg.svd(centered)
            normal = Vt[-1]  # 最小奇异值对应的奇异向量
            d = np.dot(normal, centroid)
            
            # 确保 d >= 0
            if d < 0:
                normal = -normal
                d = -d
            
            planes.append((normal, d))
            
            # 移除内点
            remaining_points = remaining_points[~best_inliers]
        
        return planes


def ransac_3d(n: int, dim: int = 3) -> RANSAC3D:
    """
    创建 RANSAC 3D 拟合器的工厂函数
    
    Args:
        n: 平面数量
        dim: 维度 (必须为 3)
        
    Returns:
        RANSAC3D 实例
    """
    return RANSAC3D(n, dim)
