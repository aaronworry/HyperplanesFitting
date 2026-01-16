#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D K-Means + PCA 平面拟合算法

使用 K-Means 聚类后对每个簇使用 PCA/SVD 拟合平面。

作者: Hyperplanes Fitting Team
"""

import numpy as np
from typing import List, Tuple


class KMeans3D:
    """
    3D K-Means + PCA 平面拟合器
    
    先使用 K-Means 将点聚类，然后对每个簇使用 SVD 拟合平面。
    """
    
    def __init__(self, n: int, dim: int = 3):
        """
        初始化 K-Means 3D 拟合器
        
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
        self.labels = None
    
    def set_data(self, data: np.ndarray):
        """
        设置数据
        
        Args:
            data: 3D 点云数据 (N x 3)
        """
        self.data = data
        self.number = len(data)
    
    def solve(self, max_iter: int = 100, tol: float = 1e-4):
        """
        求解平面拟合
        
        Args:
            max_iter: K-Means 最大迭代次数
            tol: 收敛阈值
        """
        # 1. K-Means 聚类
        self._kmeans(max_iter, tol)
        
        # 2. 对每个簇拟合平面
        for k in range(self.n):
            cluster_points = self.data[self.labels == k]
            if len(cluster_points) < 3:
                # 点数不足，使用随机平面
                self.vectors[k] = np.array([0, 0, 1])
                self.distances[k] = 0
                continue
            
            normal, d = self._fit_plane_svd(cluster_points)
            self.vectors[k] = normal
            self.distances[k] = d
    
    def _kmeans(self, max_iter: int = 100, tol: float = 1e-4):
        """
        K-Means 聚类
        
        Args:
            max_iter: 最大迭代次数
            tol: 收敛阈值
        """
        # 随机初始化聚类中心
        idx = np.random.choice(self.number, self.n, replace=False)
        centroids = self.data[idx].copy()
        
        for _ in range(max_iter):
            # 分配点到最近的聚类中心
            distances = np.zeros((self.number, self.n))
            for k in range(self.n):
                distances[:, k] = np.linalg.norm(self.data - centroids[k], axis=1)
            self.labels = np.argmin(distances, axis=1)
            
            # 更新聚类中心
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n):
                cluster_points = self.data[self.labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[k] = centroids[k]
            
            # 检查收敛
            if np.max(np.linalg.norm(new_centroids - centroids, axis=1)) < tol:
                break
            
            centroids = new_centroids
    
    def _fit_plane_svd(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        使用 SVD 拟合平面
        
        Args:
            points: 3D 点 (M x 3)
            
        Returns:
            (normal, d): 平面法向量和到原点的距离
        """
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # SVD 分解
        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1]  # 最小奇异值对应的右奇异向量
        
        # 计算 d
        d = np.dot(normal, centroid)
        
        # 确保 d >= 0
        if d < 0:
            normal = -normal
            d = -d
        
        return normal, d


def kmeans_3d(n: int, dim: int = 3) -> KMeans3D:
    """
    创建 K-Means 3D 拟合器的工厂函数
    
    Args:
        n: 平面数量
        dim: 维度 (必须为 3)
        
    Returns:
        KMeans3D 实例
    """
    return KMeans3D(n, dim)
