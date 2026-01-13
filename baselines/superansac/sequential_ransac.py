"""
SupeRANSAC 2D 直线拟合器

由于 SupeRANSAC 原生不支持 2D 直线拟合，
这里实现一个纯 Python 的 RANSAC 2D 直线拟合器，
模仿 SupeRANSAC 的接口和策略。
"""

import numpy as np
from typing import Tuple, Optional, List


class RANSACConfig:
    """RANSAC 配置参数"""
    def __init__(self):
        self.inlier_threshold = 0.1  # 内点阈值
        self.min_iterations = 100    # 最小迭代次数
        self.max_iterations = 1000   # 最大迭代次数
        self.confidence = 0.999      # 置信度
        self.min_inliers = 5         # 最小内点数
        self.use_prosac = True       # 是否使用 PROSAC 策略
        self.scoring = 'msac'        # 评分方式: 'ransac', 'msac', 'magsac'


def fit_line_2points(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    从两个点拟合直线
    
    Args:
        p1: 第一个点 (2,)
        p2: 第二个点 (2,)
    
    Returns:
        line: (3,) 直线参数 (n1, n2, d)，满足 n1*x + n2*y = d
    """
    # 方向向量
    direction = p2 - p1
    
    # 法向量（垂直于方向向量）
    normal = np.array([-direction[1], direction[0]])
    
    # 归一化
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return np.array([0, 0, 0])
    normal = normal / norm
    
    # 计算 d
    d = np.dot(normal, p1)
    
    return np.array([normal[0], normal[1], d])


def fit_line_svd(points: np.ndarray) -> np.ndarray:
    """
    使用 SVD 从多个点拟合直线（最小二乘）
    
    Args:
        points: (N, 2) 点云
    
    Returns:
        line: (3,) 直线参数 (n1, n2, d)
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


def compute_residuals(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    """
    计算点到直线的残差（距离）
    
    Args:
        points: (N, 2) 点云
        line: (3,) 直线参数 (n1, n2, d)
    
    Returns:
        residuals: (N,) 残差
    """
    n1, n2, d = line
    residuals = np.abs(n1 * points[:, 0] + n2 * points[:, 1] - d)
    return residuals


def ransac_score(residuals: np.ndarray, threshold: float, scoring: str = 'msac') -> float:
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
        # 经典 RANSAC：计算内点数
        return np.sum(residuals < threshold)
    elif scoring == 'msac':
        # M-SAC：截断的平方误差
        truncated = np.minimum(residuals ** 2, threshold ** 2)
        return -np.sum(truncated)  # 负号使其越高越好
    elif scoring == 'magsac':
        # MAGSAC-like：使用 sigma 加权
        sigma = threshold / 1.96  # 假设 threshold 对应 95% 置信度
        weights = np.exp(-residuals ** 2 / (2 * sigma ** 2))
        return np.sum(weights)
    else:
        return np.sum(residuals < threshold)


def ransac_2d_line(
    points: np.ndarray,
    config: RANSACConfig
) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
    """
    使用 RANSAC 拟合 2D 直线
    
    Args:
        points: (N, 2) 点云
        config: RANSAC 配置
    
    Returns:
        best_line: (3,) 最佳直线参数，或 None
        inlier_mask: (N,) 内点掩码
        score: 最佳得分
    """
    N = len(points)
    
    if N < 2:
        return None, np.zeros(N, dtype=bool), 0.0
    
    best_line = None
    best_score = -float('inf')
    best_inlier_mask = np.zeros(N, dtype=bool)
    
    # 计算自适应迭代次数
    def compute_iterations(inlier_ratio: float, confidence: float, mss: int = 2) -> int:
        if inlier_ratio <= 0 or inlier_ratio >= 1:
            return config.max_iterations
        prob_no_outlier = inlier_ratio ** mss
        if prob_no_outlier <= 0:
            return config.max_iterations
        return int(np.ceil(np.log(1 - confidence) / np.log(1 - prob_no_outlier)))
    
    iterations = config.min_iterations
    iteration = 0
    
    # PROSAC: 按某种质量排序（这里用到中心的距离）
    if config.use_prosac:
        centroid = np.mean(points, axis=0)
        distances_to_center = np.linalg.norm(points - centroid, axis=1)
        sorted_indices = np.argsort(distances_to_center)
        prosac_n = min(10, N)  # 从前 prosac_n 个点开始采样
    
    while iteration < iterations and iteration < config.max_iterations:
        # 采样两个点
        if config.use_prosac and iteration < N // 2:
            # PROSAC: 逐渐扩大采样范围
            prosac_n = min(10 + iteration, N)
            sample_indices = np.random.choice(sorted_indices[:prosac_n], 2, replace=False)
        else:
            sample_indices = np.random.choice(N, 2, replace=False)
        
        p1, p2 = points[sample_indices[0]], points[sample_indices[1]]
        
        # 拟合直线
        line = fit_line_2points(p1, p2)
        
        if np.linalg.norm(line[:2]) < 1e-10:
            iteration += 1
            continue
        
        # 计算残差和得分
        residuals = compute_residuals(points, line)
        score = ransac_score(residuals, config.inlier_threshold, config.scoring)
        
        if score > best_score:
            best_score = score
            best_line = line
            best_inlier_mask = residuals < config.inlier_threshold
            
            # 更新迭代次数
            inlier_ratio = np.sum(best_inlier_mask) / N
            new_iterations = compute_iterations(inlier_ratio, config.confidence)
            iterations = max(config.min_iterations, min(new_iterations, config.max_iterations))
        
        iteration += 1
    
    # 使用所有内点重新拟合
    if best_line is not None and np.sum(best_inlier_mask) >= config.min_inliers:
        inlier_points = points[best_inlier_mask]
        best_line = fit_line_svd(inlier_points)
        
        # 重新计算内点
        residuals = compute_residuals(points, best_line)
        best_inlier_mask = residuals < config.inlier_threshold
    
    return best_line, best_inlier_mask, best_score


class SequentialRANSAC2DLine:
    """
    Sequential RANSAC 多直线拟合器
    
    策略：找到一个模型 → 移除内点 → 重复
    """
    
    def __init__(self, config: RANSACConfig = None):
        self.config = config if config is not None else RANSACConfig()
    
    def fit(
        self,
        points: np.ndarray,
        max_models: int = 10,
        min_points_ratio: float = 0.05
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        拟合多条直线
        
        Args:
            points: (N, 2) 点云
            max_models: 最大模型数
            min_points_ratio: 继续拟合的最小点数比例
        
        Returns:
            lines: 直线参数列表
            labels: (N,) 每个点的标签（-1 表示离群点）
        """
        N = len(points)
        remaining_mask = np.ones(N, dtype=bool)
        labels = -np.ones(N, dtype=int)
        lines = []
        
        min_points = max(self.config.min_inliers, int(N * min_points_ratio))
        
        for model_idx in range(max_models):
            remaining_points = points[remaining_mask]
            remaining_indices = np.where(remaining_mask)[0]
            
            if len(remaining_points) < min_points:
                break
            
            # 运行 RANSAC
            line, inlier_mask_local, score = ransac_2d_line(remaining_points, self.config)
            
            if line is None or np.sum(inlier_mask_local) < self.config.min_inliers:
                break
            
            # 更新标签和剩余点
            inlier_indices = remaining_indices[inlier_mask_local]
            labels[inlier_indices] = model_idx
            remaining_mask[inlier_indices] = False
            lines.append(line)
        
        return lines, labels
    
    def fit_known_count(
        self,
        points: np.ndarray,
        num_models: int
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        拟合已知数量的直线
        
        Args:
            points: (N, 2) 点云
            num_models: 模型数量
        
        Returns:
            lines: 直线参数列表
            labels: (N,) 每个点的标签
        """
        return self.fit(points, max_models=num_models, min_points_ratio=0.01)


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt
    
    # 生成测试数据：两条直线
    np.random.seed(42)
    
    # 直线1: y = x
    t1 = np.linspace(-2, 2, 30)
    line1_points = np.column_stack([t1, t1 + np.random.randn(30) * 0.05])
    
    # 直线2: y = -x + 1
    t2 = np.linspace(-2, 2, 30)
    line2_points = np.column_stack([t2, -t2 + 1 + np.random.randn(30) * 0.05])
    
    # 合并
    points = np.vstack([line1_points, line2_points])
    
    # 运行 Sequential RANSAC
    config = RANSACConfig()
    config.inlier_threshold = 0.15
    
    seq_ransac = SequentialRANSAC2DLine(config)
    lines, labels = seq_ransac.fit(points, max_models=5)
    
    print(f"找到 {len(lines)} 条直线")
    for i, line in enumerate(lines):
        print(f"  直线 {i}: n=({line[0]:.4f}, {line[1]:.4f}), d={line[2]:.4f}")
    
    # 可视化
    plt.figure(figsize=(10, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    
    for i in range(len(lines)):
        mask = labels == i
        plt.scatter(points[mask, 0], points[mask, 1], 
                   c=colors[i % len(colors)], label=f'Line {i}', alpha=0.6)
    
    # 画离群点
    mask = labels == -1
    if np.any(mask):
        plt.scatter(points[mask, 0], points[mask, 1], c='gray', label='Outliers', alpha=0.3)
    
    # 画直线
    x_range = np.array([-3, 3])
    for i, line in enumerate(lines):
        n1, n2, d = line
        if abs(n2) > abs(n1):
            y_range = (d - n1 * x_range) / n2
            plt.plot(x_range, y_range, colors[i % len(colors)] + '--', linewidth=2)
        else:
            y_range = np.array([-3, 3])
            x_plot = (d - n2 * y_range) / n1
            plt.plot(x_plot, y_range, colors[i % len(colors)] + '--', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sequential RANSAC 2D Line Fitting')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('sequential_ransac_test.png', dpi=150)
    print("测试图像已保存到 sequential_ransac_test.png")
