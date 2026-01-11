import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression


class ransac():
    def __init__(self, ture_num, dim):
        self.dim = dim
        self.true_num = ture_num
        self.vectors = None
        self.distances = None
        
        self.data = None
        self.number = 0
        
    def set_data(self, data):
        self.data = data
        self.number = len(data)
        
        
    def solve(self):
        lines = self.get_lines()
        self.n = len(lines)
        self.vectors = np.zeros((self.n, self.dim))
        self.distances = np.array([0.]*self.n)

        for i in range(self.n):
            b, k = lines[i]
            norm_v = np.sqrt(k**2 + 1)
            self.distances[i] = -b / norm_v
            self.vectors[i, 0] = k / norm_v
            self.vectors[i, 1] = 1. / norm_v

    def get_lines(self, min_points_left = 10, min_inliers = 15):
        remaining_points = self.data
        remaining_indices = np.arange(self.number)
        all_lines = []
        
        while len(remaining_points) >= min_points_left and len(all_lines) < self.true_num:
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                max_trials=1000,
                min_samples=1.,
                residual_threshold=0.4  # 根据数据噪声调整，合理区分内点
            )
            
            # 2. 拟合当前剩余数据
            try:
                ransac.fit(remaining_points[:, 0].reshape(-1, 1), remaining_points[:, 1])
            except:
                # 数据无法拟合有效模型时，终止当前迭代
                print(f"第{len(all_lines)+1}次迭代无法拟合有效模型，终止迭代")
                break
                
            inlier_mask = ransac.inlier_mask_
            b = ransac.estimator_.intercept_
            k = ransac.estimator_.coef_[0]
            
            if len(inlier_mask) >= min_inliers:
                all_lines.append([b, k])
                remaining_points = remaining_points[~inlier_mask]
                remaining_indices = remaining_indices[~inlier_mask]
            else:
                break
        return all_lines

