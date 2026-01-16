"""
SupeRANSAC 3D 平面拟合器 (SequentialRANSAC3DPlane)

================================================================================
                        3D 平面拟合扩展说明
================================================================================

【从 2D 直线到 3D 平面的扩展】
- 2D: 最小采样集 = 2 个点 → 1 条直线
- 3D: 最小采样集 = 3 个点 → 1 个平面

【主要修改】
1. 采样: 从 2 点改为 3 点
2. 模型估计: 使用叉积计算平面法向量
3. 残差: |n1*x + n2*y + n3*z - d|

================================================================================
"""

import numpy as np
from typing import Tuple, Optional, List


class RANSACConfig3D:
    """RANSAC 3D 配置参数"""
    def __init__(self):
        self.inlier_threshold = 0.15  # 内点阈值
        self.min_iterations = 100     # 最小迭代次数
        self.max_iterations = 1000    # 最大迭代次数
        self.confidence = 0.999       # 置信度
        self.min_inliers = 10         # 最小内点数
        self.use_prosac = True        # 是否使用 PROSAC 策略
        self.scoring = 'msac'         # 评分方式: 'ransac', 'msac', 'magsac'


def fit_plane_3points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """
    从三个点拟合平面
    
    Args:
        p1, p2, p3: 三个点 (3,)
    
    Returns:
        plane: (4,) 平面参数 (n1, n2, n3, d)，满足 n1*x + n2*y + n3*z = d
    """
    # 两个方向向量
    v1 = p2 - p1
    v2 = p3 - p1
    
    # 法向量（叉积）
    normal = np.cross(v1, v2)
    
    # 归一化
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return np.array([0, 0, 0, 0])  # 三点共线
    normal = normal / norm
    
    # 计算 d
    d = np.dot(normal, p1)
    
    return np.array([normal[0], normal[1], normal[2], d])


def fit_plane_svd(points: np.ndarray) -> np.ndarray:
    """
    使用 SVD 从多个点拟合平面（最小二乘）
    
    Args:
        points: (N, 3) 点云
    
    Returns:
        plane: (4,) 平面参数 (n1, n2, n3, d)
    """
    if len(points) < 3:
        return np.array([0, 0, 0, 0])
    
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
    
    return np.array([normal[0], normal[1], normal[2], d])


def compute_residuals_3d(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """
    计算点到平面的残差（距离）
    
    Args:
        points: (N, 3) 点云
        plane: (4,) 平面参数 (n1, n2, n3, d)
    
    Returns:
        residuals: (N,) 残差
    """
    n = plane[:3]
    d = plane[3]
    residuals = np.abs(np.dot(points, n) - d)
    return residuals


def ransac_score_3d(residuals: np.ndarray, threshold: float, scoring: str = 'msac') -> float:
    """
    计算 RANSAC 得分
    
    Args:
        residuals: 残差
        threshold: 内点阈值
        scoring: 评分方式
    
    Returns:
        score: 得分（越高越好）
    """
    if scoring == 'ransac':
        return np.sum(residuals < threshold)
    elif scoring == 'msac':
        truncated = np.minimum(residuals ** 2, threshold ** 2)
        return -np.sum(truncated)
    elif scoring == 'magsac':
        sigma = threshold / 1.96
        weights = np.exp(-residuals ** 2 / (2 * sigma ** 2))
        return np.sum(weights)
    else:
        return np.sum(residuals < threshold)


def ransac_3d_plane(
    points: np.ndarray,
    config: RANSACConfig3D
) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
    """
    使用 RANSAC 拟合 3D 平面
    
    Args:
        points: (N, 3) 点云
        config: RANSAC 配置
    
    Returns:
        best_plane: (4,) 最佳平面参数，或 None
        inlier_mask: (N,) 内点掩码
        score: 最佳得分
    """
    N = len(points)
    
    if N < 3:
        return None, np.zeros(N, dtype=bool), 0.0
    
    best_plane = None
    best_score = -float('inf')
    best_inlier_mask = np.zeros(N, dtype=bool)
    
    # 计算自适应迭代次数
    def compute_iterations(inlier_ratio: float, confidence: float, mss: int = 3) -> int:
        if inlier_ratio <= 0 or inlier_ratio >= 1:
            return config.max_iterations
        prob_no_outlier = inlier_ratio ** mss
        if prob_no_outlier <= 0:
            return config.max_iterations
        return int(np.ceil(np.log(1 - confidence) / np.log(1 - prob_no_outlier)))
    
    iterations = config.min_iterations
    iteration = 0
    
    # PROSAC: 按某种质量排序
    if config.use_prosac:
        centroid = np.mean(points, axis=0)
        distances_to_center = np.linalg.norm(points - centroid, axis=1)
        sorted_indices = np.argsort(distances_to_center)
        prosac_n = min(15, N)
    
    while iteration < iterations and iteration < config.max_iterations:
        # 采样三个点
        if config.use_prosac and iteration < N // 2:
            prosac_n = min(15 + iteration, N)
            sample_indices = np.random.choice(sorted_indices[:prosac_n], 3, replace=False)
        else:
            sample_indices = np.random.choice(N, 3, replace=False)
        
        p1, p2, p3 = points[sample_indices[0]], points[sample_indices[1]], points[sample_indices[2]]
        
        # 拟合平面
        plane = fit_plane_3points(p1, p2, p3)
        
        if np.linalg.norm(plane[:3]) < 1e-10:
            iteration += 1
            continue
        
        # 计算残差和得分
        residuals = compute_residuals_3d(points, plane)
        score = ransac_score_3d(residuals, config.inlier_threshold, config.scoring)
        
        if score > best_score:
            best_score = score
            best_plane = plane
            best_inlier_mask = residuals < config.inlier_threshold
            
            # 更新迭代次数
            inlier_ratio = np.sum(best_inlier_mask) / N
            new_iterations = compute_iterations(inlier_ratio, config.confidence)
            iterations = max(config.min_iterations, min(new_iterations, config.max_iterations))
        
        iteration += 1
    
    # 使用所有内点重新拟合
    if best_plane is not None and np.sum(best_inlier_mask) >= config.min_inliers:
        inlier_points = points[best_inlier_mask]
        best_plane = fit_plane_svd(inlier_points)
        
        # 重新计算内点
        residuals = compute_residuals_3d(points, best_plane)
        best_inlier_mask = residuals < config.inlier_threshold
    
    return best_plane, best_inlier_mask, best_score


class SequentialRANSAC3DPlane:
    """
    Sequential RANSAC 多平面拟合器 (3D)
    
    策略：找到一个模型 → 移除内点 → 重复
    """
    
    def __init__(self, config: RANSACConfig3D = None):
        self.config = config if config is not None else RANSACConfig3D()
    
    def fit(
        self,
        points: np.ndarray,
        max_models: int = 10,
        min_points_ratio: float = 0.05
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        拟合多个平面
        
        Args:
            points: (N, 3) 点云
            max_models: 最大模型数
            min_points_ratio: 继续拟合的最小点数比例
        
        Returns:
            planes: 平面参数列表
            labels: (N,) 每个点的标签（-1 表示离群点）
        """
        N = len(points)
        remaining_mask = np.ones(N, dtype=bool)
        labels = -np.ones(N, dtype=int)
        planes = []
        
        min_points = max(self.config.min_inliers, int(N * min_points_ratio))
        
        for model_idx in range(max_models):
            remaining_points = points[remaining_mask]
            remaining_indices = np.where(remaining_mask)[0]
            
            if len(remaining_points) < min_points:
                break
            
            # 运行 RANSAC
            plane, inlier_mask_local, score = ransac_3d_plane(remaining_points, self.config)
            
            if plane is None or np.sum(inlier_mask_local) < self.config.min_inliers:
                break
            
            # 更新标签和剩余点
            inlier_indices = remaining_indices[inlier_mask_local]
            labels[inlier_indices] = model_idx
            remaining_mask[inlier_indices] = False
            planes.append(plane)
        
        return planes, labels
    
    def fit_known_count(
        self,
        points: np.ndarray,
        num_models: int
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        拟合已知数量的平面
        
        Args:
            points: (N, 3) 点云
            num_models: 模型数量
        
        Returns:
            planes: 平面参数列表
            labels: (N,) 每个点的标签
        """
        return self.fit(points, max_models=num_models, min_points_ratio=0.01)


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    
    # 生成测试数据: 3 个平面
    all_points = []
    
    # 平面 1: z = 0 (xy 平面)
    xy_plane_points = np.column_stack([
        np.random.uniform(-3, 3, 40),
        np.random.uniform(-3, 3, 40),
        np.random.randn(40) * 0.05
    ])
    all_points.append(xy_plane_points)
    
    # 平面 2: y = 0 (xz 平面)
    xz_plane_points = np.column_stack([
        np.random.uniform(-3, 3, 40),
        np.random.randn(40) * 0.05,
        np.random.uniform(-3, 3, 40)
    ])
    all_points.append(xz_plane_points)
    
    # 平面 3: x = 0 (yz 平面)
    yz_plane_points = np.column_stack([
        np.random.randn(40) * 0.05,
        np.random.uniform(-3, 3, 40),
        np.random.uniform(-3, 3, 40)
    ])
    all_points.append(yz_plane_points)
    
    points = np.vstack(all_points)
    
    # 运行 Sequential RANSAC
    config = RANSACConfig3D()
    config.inlier_threshold = 0.15
    
    seq_ransac = SequentialRANSAC3DPlane(config)
    planes, labels = seq_ransac.fit(points, max_models=5)
    
    print(f"找到 {len(planes)} 个平面")
    for i, plane in enumerate(planes):
        print(f"  平面 {i}: n=({plane[0]:.4f}, {plane[1]:.4f}, {plane[2]:.4f}), d={plane[3]:.4f}")
    
    print(f"\n聚类标签分布: {np.bincount(labels[labels >= 0])}")
