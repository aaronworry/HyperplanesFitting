#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D GMM (高斯混合模型) 平面拟合算法

使用 GMM 进行软聚类后对每个簇使用加权 PCA/SVD 拟合平面。

作者: Hyperplanes Fitting Team
"""

import numpy as np
from typing import List, Tuple
from scipy.stats import multivariate_normal


class GMM3D:
    """
    3D GMM + PCA 平面拟合器
    
    使用高斯混合模型进行软聚类，然后对每个簇使用加权 SVD 拟合平面。
    """
    
    def __init__(self, n: int, dim: int = 3):
        """
        初始化 GMM 3D 拟合器
        
        Args:
            n: 要拟合的平面数量/高斯分量数
            dim: 数据维度 (必须为 3)
        """
        self.n = n
        self.dim = dim
        self.vectors = np.zeros((n, dim))
        self.distances = np.zeros(n)
        
        self.data = None
        self.number = 0
        
        # GMM 参数
        self.mu = None          # 均值
        self.sigma = None       # 协方差矩阵
        self.pi = None          # 混合系数
        self.weights = None     # 后验概率
    
    def set_data(self, data: np.ndarray):
        """
        设置数据
        
        Args:
            data: 3D 点云数据 (N x 3)
        """
        self.data = data
        self.number = len(data)
        
        # 初始化 GMM 参数
        idx = np.random.choice(self.number, self.n, replace=False)
        self.mu = self.data[idx].copy()
        self.sigma = np.array([2.0 * np.eye(self.dim) for _ in range(self.n)])
        self.pi = np.ones(self.n) / self.n
        self.weights = np.ones((self.number, self.n)) / self.n
    
    def solve(self, max_iter: int = 50, tol: float = 1e-4):
        """
        求解平面拟合
        
        Args:
            max_iter: EM 最大迭代次数
            tol: 收敛阈值
        """
        # 1. EM 算法拟合 GMM
        self._fit_gmm(max_iter, tol)
        
        # 2. 对每个分量拟合平面
        for k in range(self.n):
            # 获取该分量的点的权重
            cluster_weights = self.weights[:, k]
            
            if np.sum(cluster_weights) < 1e-10:
                self.vectors[k] = np.array([0, 0, 1])
                self.distances[k] = 0
                continue
            
            normal, d = self._fit_plane_weighted_svd(self.data, cluster_weights)
            self.vectors[k] = normal
            self.distances[k] = d
    
    def _fit_gmm(self, max_iter: int = 50, tol: float = 1e-4):
        """
        使用 EM 算法拟合 GMM
        """
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iter):
            # E 步: 计算后验概率
            self._e_step()
            
            # M 步: 更新参数
            self._m_step()
            
            # 计算对数似然
            log_likelihood = self._compute_log_likelihood()
            
            # 检查收敛
            if abs(log_likelihood - prev_log_likelihood) < tol:
                break
            
            prev_log_likelihood = log_likelihood
    
    def _e_step(self):
        """E 步: 计算后验概率"""
        pdfs = np.zeros((self.number, self.n))
        
        for k in range(self.n):
            try:
                pdfs[:, k] = self.pi[k] * multivariate_normal.pdf(
                    self.data, self.mu[k], self.sigma[k], allow_singular=True
                )
            except:
                pdfs[:, k] = 1e-10
        
        # 归一化
        row_sums = pdfs.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-10] = 1e-10
        self.weights = pdfs / row_sums
    
    def _m_step(self):
        """M 步: 更新参数"""
        Nk = self.weights.sum(axis=0)
        
        for k in range(self.n):
            if Nk[k] < 1e-10:
                continue
            
            # 更新均值
            self.mu[k] = np.average(self.data, axis=0, weights=self.weights[:, k])
            
            # 更新协方差
            diff = self.data - self.mu[k]
            weighted_diff = diff * self.weights[:, k:k+1]
            self.sigma[k] = np.dot(weighted_diff.T, diff) / Nk[k]
            
            # 添加正则化项防止奇异
            self.sigma[k] += 1e-6 * np.eye(self.dim)
        
        # 更新混合系数
        self.pi = Nk / self.number
    
    def _compute_log_likelihood(self) -> float:
        """计算对数似然"""
        pdfs = np.zeros((self.number, self.n))
        
        for k in range(self.n):
            try:
                pdfs[:, k] = self.pi[k] * multivariate_normal.pdf(
                    self.data, self.mu[k], self.sigma[k], allow_singular=True
                )
            except:
                pdfs[:, k] = 1e-10
        
        return np.sum(np.log(np.maximum(pdfs.sum(axis=1), 1e-10)))
    
    def _fit_plane_weighted_svd(self, points: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        使用加权 SVD 拟合平面
        
        Args:
            points: 3D 点 (N x 3)
            weights: 点权重 (N,)
            
        Returns:
            (normal, d): 平面法向量和到原点的距离
        """
        # 加权中心
        weights = weights / np.sum(weights)
        centroid = np.average(points, axis=0, weights=weights)
        
        # 中心化
        centered = points - centroid
        
        # 加权协方差矩阵
        weighted_cov = np.dot((centered * weights[:, np.newaxis]).T, centered)
        
        # 特征分解
        eigenvalues, eigenvectors = np.linalg.eigh(weighted_cov)
        
        # 最小特征值对应的特征向量即为法向量
        normal = eigenvectors[:, 0]
        
        # 计算 d
        d = np.dot(normal, centroid)
        
        # 确保 d >= 0
        if d < 0:
            normal = -normal
            d = -d
        
        return normal, d


def gmm_3d(n: int, dim: int = 3) -> GMM3D:
    """
    创建 GMM 3D 拟合器的工厂函数
    
    Args:
        n: 分量数量/平面数量
        dim: 维度 (必须为 3)
        
    Returns:
        GMM3D 实例
    """
    return GMM3D(n, dim)
