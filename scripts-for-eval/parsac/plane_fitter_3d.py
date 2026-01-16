"""
PARSAC 多平面拟合适配器 (SimplePARSACPlaneFitter) - 3D 版本

================================================================================
                        3D 平面拟合扩展说明
================================================================================

【从 2D 直线到 3D 平面的扩展】
- 2D: 最小采样集 = 2 个点 → 1 条直线 (n1*x + n2*y = d)
- 3D: 最小采样集 = 3 个点 → 1 个平面 (n1*x + n2*y + n3*z = d)

【主要修改】
1. 假设生成: 从 3 个点计算平面法向量 n = (p2-p1) × (p3-p1)
2. 残差计算: |n1*x + n2*y + n3*z - d|
3. SVD 精化: 在 3D 点云上使用 SVD 拟合平面

================================================================================
"""

import numpy as np
from typing import Tuple, List, Optional


class SimplePARSACPlaneFitter:
    """
    简化版 PARSAC 多平面拟合器 (3D)
    
    与 2D 版本 (SimplePARSACLineFitter) 的区别:
    - 从 3 个点生成平面假设
    - 法向量在 R^3 中
    - 支持 3D 点云输入
    """
    
    def __init__(
        self,
        num_hypotheses: int = 200,
        num_instances: int = 4,
        inlier_threshold: float = 0.15,
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
        生成候选平面假设
        
        Args:
            points: (N, 3) 点云
        
        Returns:
            hypotheses: (H, 4) 候选平面 [n1, n2, n3, d]
        """
        N = len(points)
        hypotheses = []
        
        for _ in range(self.num_hypotheses):
            # 随机采样三个点
            idx = np.random.choice(N, 3, replace=False)
            p1, p2, p3 = points[idx[0]], points[idx[1]], points[idx[2]]
            
            # 计算平面法向量: 两个向量的叉积
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue  # 三点共线，跳过
            normal = normal / norm
            
            # 计算 d
            d = np.dot(normal, p1)
            
            hypotheses.append([normal[0], normal[1], normal[2], d])
        
        return np.array(hypotheses) if hypotheses else np.zeros((0, 4))
    
    def _compute_residuals(self, points: np.ndarray, hypotheses: np.ndarray) -> np.ndarray:
        """
        计算残差矩阵
        
        Args:
            points: (N, 3) 点云
            hypotheses: (H, 4) 候选平面 [n1, n2, n3, d]
        
        Returns:
            residuals: (N, H) 残差矩阵
        """
        N = len(points)
        H = len(hypotheses)
        
        if H == 0:
            return np.zeros((N, 0))
        
        # 向量化计算: residual = |n · x - d|
        # hypotheses[:, :3] 是法向量 (H, 3)
        # hypotheses[:, 3] 是 d (H,)
        # points 是 (N, 3)
        
        # (N, H) = (N, 3) @ (3, H) - (H,)
        residuals = np.abs(points @ hypotheses[:, :3].T - hypotheses[:, 3])
        
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
            hypotheses: (H, 4) 候选平面
            inlier_scores: (N, H) 软内点得分
            num_select: 选择数量
        
        Returns:
            selected: (K, 4) 选择的平面
        """
        if len(hypotheses) == 0:
            return np.zeros((0, 4))
        
        # 计算每个假设的总得分
        total_scores = np.sum(inlier_scores, axis=0)
        
        # 贪心选择
        selected_indices = []
        selected_planes = []
        assigned_points = np.zeros(inlier_scores.shape[0], dtype=bool)
        
        for _ in range(min(num_select, len(hypotheses))):
            if len(selected_indices) == 0:
                best_idx = np.argmax(total_scores)
            else:
                # 惩罚与已选平面相似的
                similarity_penalty = np.zeros(len(hypotheses))
                for idx in selected_indices:
                    # 使用法向量夹角作为相似度
                    normals = hypotheses[:, :3]
                    ref_normal = hypotheses[idx, :3]
                    cosines = np.abs(np.sum(normals * ref_normal, axis=1))
                    
                    # 使用 d 的差异
                    d_diff = np.abs(hypotheses[:, 3] - hypotheses[idx, 3])
                    
                    # 如果方向相同且 d 相近，则惩罚
                    same_plane = (cosines > 0.95) & (d_diff < 0.5)
                    similarity_penalty = np.maximum(similarity_penalty, same_plane.astype(float) * 0.9)
                    similarity_penalty = np.maximum(similarity_penalty, cosines * 0.3)
                
                # 重新计算得分，只考虑未分配的点
                remaining_scores = np.sum(inlier_scores[~assigned_points, :], axis=0)
                adjusted_scores = remaining_scores * (1 - similarity_penalty)
                adjusted_scores[selected_indices] = -np.inf
                best_idx = np.argmax(adjusted_scores)
            
            if total_scores[best_idx] <= 0:
                break
            
            selected_indices.append(best_idx)
            selected_planes.append(hypotheses[best_idx])
            
            # 标记这个平面的内点为已分配
            hard_inliers = inlier_scores[:, best_idx] > 0.5
            assigned_points = assigned_points | hard_inliers
        
        return np.array(selected_planes) if selected_planes else np.zeros((0, 4))
    
    def _cluster_points(
        self,
        points: np.ndarray,
        planes: np.ndarray
    ) -> np.ndarray:
        """
        将点聚类到最近的平面
        
        Args:
            points: (N, 3) 点云
            planes: (K, 4) 平面
        
        Returns:
            labels: (N,) 聚类标签
        """
        N = len(points)
        if len(planes) == 0:
            return -np.ones(N, dtype=int)
        
        residuals = self._compute_residuals(points, planes)
        labels = np.argmin(residuals, axis=1)
        
        return labels
    
    def _refine_planes(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        num_planes: int
    ) -> np.ndarray:
        """
        使用聚类结果重新拟合平面
        
        Args:
            points: (N, 3) 点云
            labels: (N,) 聚类标签
            num_planes: 平面数量
        
        Returns:
            refined_planes: (K, 4) 精化后的平面
        """
        refined_planes = []
        
        for k in range(num_planes):
            mask = labels == k
            if np.sum(mask) < 3:
                continue
            
            cluster_points = points[mask]
            
            # SVD 拟合
            centroid = np.mean(cluster_points, axis=0)
            centered = cluster_points - centroid
            U, S, Vt = np.linalg.svd(centered)
            
            # 法向量是最小奇异值对应的右奇异向量
            normal = Vt[-1]
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
            
            d = np.dot(normal, centroid)
            refined_planes.append([normal[0], normal[1], normal[2], d])
        
        return np.array(refined_planes) if refined_planes else np.zeros((0, 4))
    
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
            points: (N, 3) 点云
            hypotheses: (H, 4) 候选平面
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
        拟合多个平面
        
        Args:
            points: (N, 3) 点云
            num_models: 模型数量（None 且 auto_detect=True 时自动检测，
                        否则使用 self.num_instances）
            num_iterations: 迭代次数
            auto_detect: 是否自动检测模型数量（当 num_models=None 时生效）
        
        Returns:
            planes: (K, 4) 平面参数 [n1, n2, n3, d]
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
        
        best_planes = None
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
            selected_planes = self._select_hypotheses(hypotheses, inlier_scores, num_models)
            
            if len(selected_planes) == 0:
                continue
            
            # 4. 聚类点
            labels = self._cluster_points(points, selected_planes)
            
            # 5. 精化平面
            refined_planes = self._refine_planes(points, labels, len(selected_planes))
            
            # 6. 重新聚类
            if len(refined_planes) > 0:
                labels = self._cluster_points(points, refined_planes)
                
                # 计算总代价
                final_residuals = self._compute_residuals(points, refined_planes)
                min_residuals = np.min(final_residuals, axis=1)
                total_cost = np.sum(min_residuals)
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_planes = refined_planes
                    best_labels = labels
        
        if best_planes is None:
            return np.zeros((0, 4)), -np.ones(len(points), dtype=int)
        
        return best_planes, best_labels


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
    
    # 平面 3: x + y + z = 3
    t1 = np.random.uniform(-2, 2, 40)
    t2 = np.random.uniform(-2, 2, 40)
    plane3_points = np.column_stack([
        t1,
        t2,
        3 - t1 - t2 + np.random.randn(40) * 0.05
    ])
    all_points.append(plane3_points)
    
    points = np.vstack(all_points)
    
    # 运行拟合
    fitter = SimplePARSACPlaneFitter(num_hypotheses=300, inlier_threshold=0.15)
    planes, labels = fitter.fit(points, num_models=3)
    
    print(f"找到 {len(planes)} 个平面")
    for i, plane in enumerate(planes):
        print(f"  平面 {i}: n=({plane[0]:.4f}, {plane[1]:.4f}, {plane[2]:.4f}), d={plane[3]:.4f}")
    
    print(f"\n聚类标签分布: {np.bincount(labels[labels >= 0])}")
