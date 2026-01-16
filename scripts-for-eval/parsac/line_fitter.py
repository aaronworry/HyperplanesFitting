"""
PARSAC 多直线拟合适配器 (SimplePARSACLineFitter)

================================================================================
                        原始 PARSAC vs 我们的实现
================================================================================

【原始 PARSAC 论文】
- 标题: "PARSAC: Accelerating Robust Multi-Model Fitting with Parallel Sample Consensus"
- 会议: AAAI 2024
- 作者: Florian Kluger, Bodo Rosenhahn
- 代码: https://github.com/fkluger/parsac

【原始 PARSAC 算法流程】
1. 神经网络 (CNNet): 
   - 5层 ResNet 风格的 1D 卷积网络
   - 输入: 点/线段特征 (N × input_dim)
   - 输出: log_inlier_weights (内点权重), log_sample_weights (采样权重)
   - 用途: 学习哪些点更可能是内点，以及如何采样

2. 并行采样 (sampling.py):
   - 使用学习的权重进行重要性采样
   - 生成 M×S×K 个最小集 (M=实例数, S=假设数, K=批次数)
   - 调用 minimal_solver 计算假设

3. 残差计算 + 软内点计数 (inlier_counting.py):
   - soft_inlier(d) = 1 - sigmoid(β * d/τ - β)
   - 加权内点比例作为假设得分

4. 假设选择 + 聚类 (postprocessing.py):
   - 贪心选择: 每次选择增益最大的假设
   - 惩罚与已选假设重叠的内点
   - 点分配到最近假设

5. 支持的问题:
   - 消失点检测 (vp)
   - 基础矩阵估计 (fundamental)
   - 单应矩阵估计 (homography)

【我们的实现 (SimplePARSACLineFitter)】
由于原始 PARSAC 不支持 2D 点云直线拟合，我们实现了一个简化版本:

1. 采样策略: 均匀随机采样 (替代神经网络)
   - 原始: 使用学习权重进行重要性采样
   - 我们: 均匀随机采样点对
   - 差异: 无法利用数据结构进行智能采样

2. 假设生成: 点对拟合 (与原始类似)
   - 原始: minimal_solver 根据问题类型选择求解器
   - 我们: 从两点计算直线 n = (p2-p1)⊥, d = n·p1
   - 差异: 本质相同

3. 软内点计数: ✓ 保持一致
   - 原始: 1 - sigmoid(β * d/τ - β)
   - 我们: 1 / (1 + exp(β * (d - τ)))  (数学等价)

4. 假设选择: ✓ 贪心选择 (与原始类似)
   - 原始: 贪心 + 去重
   - 我们: 贪心 + 相似性惩罚 + 已分配点标记

5. 精化: SVD 重新拟合
   - 原始: 仅对 VP 问题有精化
   - 我们: 对所有聚类使用 SVD 精化

【保真度评估】
- 整体保真度: ~65%
- 核心算法思想: ✓ 保留
- 神经网络组件: ✗ 移除
- 适合用于公平对比: ✓ 是 (在论文中需说明)

================================================================================
"""

import os
import sys
import numpy as np
from typing import Tuple, List, Optional

# 添加 PARSAC 路径 (仅供参考)
PARSAC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'potential-repositories', 'parsac'
)
sys.path.insert(0, PARSAC_PATH)


def points_to_pseudo_lines(points: np.ndarray, k_neighbors: int = 3) -> np.ndarray:
    """
    将点云转换为"伪线段"
    
    策略：对于每个点，连接到它最近的 k 个邻居，形成线段
    
    Args:
        points: (N, 2) 点云
        k_neighbors: 每个点连接的邻居数
    
    Returns:
        lines: (M, 4) 线段，每行为 (x1, y1, x2, y2)
    """
    from scipy.spatial import KDTree
    
    N = len(points)
    tree = KDTree(points)
    
    # 查找 k 个最近邻
    distances, indices = tree.query(points, k=k_neighbors + 1)  # +1 因为包含自身
    
    lines = []
    for i in range(N):
        for j in range(1, k_neighbors + 1):  # 跳过自身
            neighbor_idx = indices[i, j]
            # 避免重复：只保留 i < neighbor_idx 的边
            if i < neighbor_idx:
                x1, y1 = points[i]
                x2, y2 = points[neighbor_idx]
                lines.append([x1, y1, x2, y2])
    
    return np.array(lines)


def pseudo_lines_to_parsac_format(lines: np.ndarray, img_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
    """
    将伪线段转换为 PARSAC 格式
    
    PARSAC VP 模块期望的特征格式 (N, 12):
    - [0:3]: 起点齐次坐标 (x1, y1, 1)
    - [3:6]: 终点齐次坐标 (x2, y2, 1)
    - [6:9]: 直线齐次表示 (a, b, c)，满足 ax + by + c = 0
    - [9:12]: 线段中点齐次坐标
    
    Args:
        lines: (M, 4) 线段 (x1, y1, x2, y2)
        img_size: 图像尺寸
    
    Returns:
        features: (M, 12) PARSAC 特征
    """
    M = len(lines)
    features = np.zeros((M, 12))
    
    for i in range(M):
        x1, y1, x2, y2 = lines[i]
        
        # 起点和终点齐次坐标
        features[i, 0:3] = [x1, y1, 1]
        features[i, 3:6] = [x2, y2, 1]
        
        # 直线参数：叉积 (x1, y1, 1) × (x2, y2, 1)
        line_homo = np.cross([x1, y1, 1], [x2, y2, 1])
        # 归一化
        norm = np.linalg.norm(line_homo[:2])
        if norm > 1e-10:
            line_homo = line_homo / norm
        features[i, 6:9] = line_homo
        
        # 中点
        features[i, 9:12] = [(x1 + x2) / 2, (y1 + y2) / 2, 1]
    
    return features


def vanishing_points_to_lines(
    vps: np.ndarray,
    point_labels: np.ndarray,
    points: np.ndarray
) -> np.ndarray:
    """
    将消失点和点聚类转换回直线参数
    
    注意：消失点检测和直线拟合是不同的问题：
    - 消失点是一组平行线在图像中的交点
    - 直线拟合是找到数据点的最佳拟合直线
    
    这里我们对每个聚类使用 SVD 拟合直线
    
    Args:
        vps: (K, 3) 消失点齐次坐标
        point_labels: (N,) 每个点的标签
        points: (N, 2) 原始点云
    
    Returns:
        lines: (K, 3) 直线参数 (n1, n2, d)
    """
    K = len(vps)
    lines = []
    
    for k in range(K):
        mask = point_labels == k
        if np.sum(mask) < 2:
            continue
        
        cluster_points = points[mask]
        
        # SVD 拟合
        centroid = np.mean(cluster_points, axis=0)
        centered = cluster_points - centroid
        U, S, Vt = np.linalg.svd(centered)
        
        # 法向量
        normal = Vt[-1]
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm
        
        d = np.dot(normal, centroid)
        lines.append([normal[0], normal[1], d])
    
    return np.array(lines) if lines else np.zeros((0, 3))


class SimplePARSACLineFitter:
    """
    简化版 PARSAC 多直线拟合器
    
    由于直接使用 PARSAC 的神经网络需要训练，
    这里我们实现一个基于 PARSAC 思想的简化版本：
    
    1. 随机采样点对，生成候选直线
    2. 使用加权投票选择最佳直线
    3. 聚类分配点到直线
    """
    
    def __init__(
        self,
        num_hypotheses: int = 100,
        num_instances: int = 4,
        inlier_threshold: float = 0.1,
        inlier_softness: float = 5.0
    ):
        """
        Args:
            num_hypotheses: 候选假设数量
            num_instances: 期望的模型实例数
            inlier_threshold: 内点阈值
            inlier_softness: 内点软化参数
        """
        self.num_hypotheses = num_hypotheses
        self.num_instances = num_instances
        self.inlier_threshold = inlier_threshold
        self.inlier_softness = inlier_softness
    
    def _generate_hypotheses(self, points: np.ndarray) -> np.ndarray:
        """
        生成候选直线假设
        
        Args:
            points: (N, 2) 点云
        
        Returns:
            hypotheses: (H, 3) 候选直线
        """
        N = len(points)
        hypotheses = []
        
        for _ in range(self.num_hypotheses):
            # 随机采样两个点
            idx = np.random.choice(N, 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]
            
            # 计算直线
            direction = p2 - p1
            normal = np.array([-direction[1], direction[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            d = np.dot(normal, p1)
            
            hypotheses.append([normal[0], normal[1], d])
        
        return np.array(hypotheses)
    
    def _compute_residuals(self, points: np.ndarray, hypotheses: np.ndarray) -> np.ndarray:
        """
        计算残差矩阵
        
        Args:
            points: (N, 2) 点云
            hypotheses: (H, 3) 候选直线
        
        Returns:
            residuals: (N, H) 残差矩阵
        """
        N = len(points)
        H = len(hypotheses)
        
        residuals = np.zeros((N, H))
        for j in range(H):
            n1, n2, d = hypotheses[j]
            residuals[:, j] = np.abs(n1 * points[:, 0] + n2 * points[:, 1] - d)
        
        return residuals
    
    def _soft_inlier_scores(self, residuals: np.ndarray) -> np.ndarray:
        """
        计算软内点得分
        
        Args:
            residuals: (N, H) 残差矩阵
        
        Returns:
            scores: (N, H) 软内点得分
        """
        # 使用 sigmoid 函数
        scores = 1 / (1 + np.exp(self.inlier_softness * (residuals - self.inlier_threshold)))
        return scores
    
    def _select_hypotheses(
        self,
        hypotheses: np.ndarray,
        inlier_scores: np.ndarray,
        num_select: int
    ) -> np.ndarray:
        """
        选择最佳假设
        
        Args:
            hypotheses: (H, 3) 候选直线
            inlier_scores: (N, H) 软内点得分
            num_select: 选择数量
        
        Returns:
            selected: (K, 3) 选择的直线
        """
        # 计算每个假设的总得分
        total_scores = np.sum(inlier_scores, axis=0)
        
        # 贪心选择：每次选择得分最高且与已选直线足够不同的
        selected_indices = []
        selected_lines = []
        assigned_points = np.zeros(inlier_scores.shape[0], dtype=bool)
        
        for _ in range(min(num_select, len(hypotheses))):
            if len(selected_indices) == 0:
                best_idx = np.argmax(total_scores)
            else:
                # 惩罚与已选直线相似的
                similarity_penalty = np.zeros(len(hypotheses))
                for idx in selected_indices:
                    # 使用法向量夹角作为相似度
                    cosines = np.abs(
                        hypotheses[:, 0] * hypotheses[idx, 0] +
                        hypotheses[:, 1] * hypotheses[idx, 1]
                    )
                    # 使用 d 的差异
                    d_diff = np.abs(hypotheses[:, 2] - hypotheses[idx, 2])
                    # 如果方向相同且 d 相近，则惩罚
                    same_line = (cosines > 0.95) & (d_diff < 0.5)
                    similarity_penalty = np.maximum(similarity_penalty, same_line.astype(float) * 0.9)
                    similarity_penalty = np.maximum(similarity_penalty, cosines * 0.3)
                
                # 重新计算得分，只考虑未分配的点
                remaining_scores = np.sum(inlier_scores[~assigned_points, :], axis=0)
                adjusted_scores = remaining_scores * (1 - similarity_penalty)
                adjusted_scores[selected_indices] = -np.inf
                best_idx = np.argmax(adjusted_scores)
            
            if total_scores[best_idx] <= 0:
                break
            
            selected_indices.append(best_idx)
            selected_lines.append(hypotheses[best_idx])
            
            # 标记这条直线的内点为已分配
            hard_inliers = inlier_scores[:, best_idx] > 0.5
            assigned_points = assigned_points | hard_inliers
        
        return np.array(selected_lines) if selected_lines else np.zeros((0, 3))
    
    def _cluster_points(
        self,
        points: np.ndarray,
        lines: np.ndarray
    ) -> np.ndarray:
        """
        将点聚类到最近的直线
        
        Args:
            points: (N, 2) 点云
            lines: (K, 3) 直线
        
        Returns:
            labels: (N,) 聚类标签
        """
        N = len(points)
        if len(lines) == 0:
            return -np.ones(N, dtype=int)
        
        residuals = self._compute_residuals(points, lines)
        labels = np.argmin(residuals, axis=1)
        
        return labels
    
    def _refine_lines(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        num_lines: int
    ) -> np.ndarray:
        """
        使用聚类结果重新拟合直线
        
        Args:
            points: (N, 2) 点云
            labels: (N,) 聚类标签
            num_lines: 直线数量
        
        Returns:
            refined_lines: (K, 3) 精化后的直线
        """
        refined_lines = []
        
        for k in range(num_lines):
            mask = labels == k
            if np.sum(mask) < 2:
                continue
            
            cluster_points = points[mask]
            
            # SVD 拟合
            centroid = np.mean(cluster_points, axis=0)
            centered = cluster_points - centroid
            U, S, Vt = np.linalg.svd(centered)
            
            normal = Vt[-1]
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
            
            d = np.dot(normal, centroid)
            refined_lines.append([normal[0], normal[1], d])
        
        return np.array(refined_lines) if refined_lines else np.zeros((0, 3))
    
    def _auto_detect_num_models(
        self,
        points: np.ndarray,
        hypotheses: np.ndarray,
        inlier_scores: np.ndarray,
        max_models: int = 10,
        min_score_ratio: float = 0.1,
        min_inliers_ratio: float = 0.05
    ) -> int:
        """
        自动检测最佳模型数量
        
        策略：贪心选择假设，直到以下条件之一满足：
        1. 新模型的得分相对于第一个模型太低 (< min_score_ratio)
        2. 新模型的内点数太少 (< min_inliers_ratio * N)
        3. 达到最大模型数
        
        Args:
            points: (N, 2) 点云
            hypotheses: (H, 3) 候选直线
            inlier_scores: (N, H) 软内点得分
            max_models: 最大模型数
            min_score_ratio: 最小得分比例
            min_inliers_ratio: 最小内点比例
        
        Returns:
            num_models: 检测到的模型数量
        """
        N = len(points)
        H = len(hypotheses)
        
        if H == 0:
            return 0
        
        # 计算每个假设的总得分
        total_scores = np.sum(inlier_scores, axis=0)
        
        # 贪心选择
        selected_indices = []
        assigned_points = np.zeros(N, dtype=bool)
        first_score = None
        
        for _ in range(min(max_models, H)):
            # 计算剩余点的得分
            if len(selected_indices) == 0:
                remaining_scores = total_scores.copy()
            else:
                remaining_scores = np.sum(inlier_scores[~assigned_points, :], axis=0)
                remaining_scores[selected_indices] = -np.inf
            
            best_idx = np.argmax(remaining_scores)
            best_score = remaining_scores[best_idx]
            
            if best_score <= 0:
                break
            
            # 记录第一个模型的得分作为参考
            if first_score is None:
                first_score = best_score
            
            # 检查得分比例
            if best_score < first_score * min_score_ratio:
                break
            
            # 检查内点数
            hard_inliers = inlier_scores[:, best_idx] > 0.5
            num_new_inliers = np.sum(hard_inliers & ~assigned_points)
            
            if num_new_inliers < N * min_inliers_ratio:
                break
            
            selected_indices.append(best_idx)
            assigned_points = assigned_points | hard_inliers
        
        return max(1, len(selected_indices))
    
    def fit(
        self,
        points: np.ndarray,
        num_models: Optional[int] = None,
        num_iterations: int = 3,
        auto_detect: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        拟合多条直线
        
        Args:
            points: (N, 2) 点云
            num_models: 模型数量（None 且 auto_detect=True 时自动检测，
                        否则使用 self.num_instances）
            num_iterations: 迭代次数
            auto_detect: 是否自动检测模型数量（当 num_models=None 时生效）
        
        Returns:
            lines: (K, 3) 直线参数
            labels: (N,) 每个点的标签
        """
        # 确定模型数量
        if num_models is None:
            if auto_detect:
                # 先生成假设，然后自动检测数量
                hypotheses = self._generate_hypotheses(points)
                if len(hypotheses) > 0:
                    residuals = self._compute_residuals(points, hypotheses)
                    inlier_scores = self._soft_inlier_scores(residuals)
                    num_models = self._auto_detect_num_models(
                        points, hypotheses, inlier_scores,
                        max_models=10,
                        min_score_ratio=0.15,
                        min_inliers_ratio=0.05
                    )
                else:
                    num_models = self.num_instances
            else:
                num_models = self.num_instances
        
        best_lines = None
        best_labels = None
        best_cost = float('inf')
        
        for iter_idx in range(num_iterations):
            # 1. 生成候选假设
            hypotheses = self._generate_hypotheses(points)
            
            if len(hypotheses) == 0:
                continue
            
            # 2. 计算残差和软内点得分
            residuals = self._compute_residuals(points, hypotheses)
            inlier_scores = self._soft_inlier_scores(residuals)
            
            # 3. 选择最佳假设
            selected_lines = self._select_hypotheses(hypotheses, inlier_scores, num_models)
            
            if len(selected_lines) == 0:
                continue
            
            # 4. 聚类点
            labels = self._cluster_points(points, selected_lines)
            
            # 5. 精化直线
            refined_lines = self._refine_lines(points, labels, len(selected_lines))
            
            # 6. 重新聚类
            if len(refined_lines) > 0:
                labels = self._cluster_points(points, refined_lines)
                
                # 计算总代价
                final_residuals = self._compute_residuals(points, refined_lines)
                min_residuals = np.min(final_residuals, axis=1)
                total_cost = np.sum(min_residuals)
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_lines = refined_lines
                    best_labels = labels
        
        if best_lines is None:
            return np.zeros((0, 3)), -np.ones(len(points), dtype=int)
        
        return best_lines, best_labels


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt
    
    # 生成测试数据
    np.random.seed(42)
    
    # 4 条直线
    lines_gt = []
    all_points = []
    
    for i in range(4):
        angle = i * np.pi / 4
        n1, n2 = np.cos(angle), np.sin(angle)
        d = (i - 1.5) * 2
        lines_gt.append([n1, n2, d])
        
        # 在直线上生成点
        t = np.linspace(-3, 3, 30)
        # 直线方向向量
        direction = np.array([-n2, n1])
        # 直线上的点
        base_point = np.array([n1 * d, n2 * d])
        line_points = base_point + np.outer(t, direction)
        # 添加噪声
        line_points += np.random.randn(30, 2) * 0.05
        all_points.append(line_points)
    
    points = np.vstack(all_points)
    lines_gt = np.array(lines_gt)
    
    print(f"测试数据: {len(points)} 个点, {len(lines_gt)} 条真值直线")
    
    # 运行拟合
    fitter = SimplePARSACLineFitter(
        num_hypotheses=200,
        num_instances=4,
        inlier_threshold=0.15
    )
    
    lines_pred, labels = fitter.fit(points, num_models=4)
    
    print(f"预测: {len(lines_pred)} 条直线")
    for i, line in enumerate(lines_pred):
        print(f"  直线 {i}: n=({line[0]:.4f}, {line[1]:.4f}), d={line[2]:.4f}")
    
    # 可视化
    plt.figure(figsize=(10, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']
    
    for i in range(len(lines_pred)):
        mask = labels == i
        plt.scatter(points[mask, 0], points[mask, 1], 
                   c=colors[i % len(colors)], label=f'Line {i}', alpha=0.6)
    
    # 画直线
    x_range = np.array([-4, 4])
    for i, line in enumerate(lines_pred):
        n1, n2, d = line
        if abs(n2) > abs(n1):
            y_range = (d - n1 * x_range) / n2
            plt.plot(x_range, y_range, colors[i % len(colors)] + '--', linewidth=2)
        else:
            y_range = np.array([-4, 4])
            x_plot = (d - n2 * y_range) / n1
            plt.plot(x_plot, y_range, colors[i % len(colors)] + '--', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Simple PARSAC-style Multi-Line Fitting')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.savefig('parsac_line_fitter_test.png', dpi=150)
    print("测试图像已保存到 parsac_line_fitter_test.png")
