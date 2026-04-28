import numpy as np
import math


class Random():
    def __init__(self, dim, hyperplane_num, max_distance_from_hyperplane = 0.5, min_points_on_hyperplane = 10, max_points_on_hyperplane = 30, X_limit=[-5., 5.], Y_limit=[-5., 5.], Z_limit=[-5., 5.], W_limit=[-5., 5.], U_limit=[-5., 5.], V_limit=[-5., 5.]):
        self.dim = dim
        self.hyperplane_num = hyperplane_num
        self.min_points_on_hyperplane = min_points_on_hyperplane
        self.max_points_on_hyperplane = max_points_on_hyperplane
        self.max_distance_from_hyperplane = max_distance_from_hyperplane
        self.xlim = X_limit
        self.ylim = Y_limit
        self.zlim = Z_limit
        self.wlim = W_limit
        self.ulim = U_limit
        self.vlim = V_limit
        self.ground_truth_A = None
        self.ground_truth_b = None

    def _coord_limits(self):
        """各坐标轴采样范围，长度与 self.dim 一致。"""
        lims = [self.xlim, self.ylim]
        if self.dim >= 3:
            lims.append(self.zlim)
        if self.dim >= 4:
            lims.append(self.wlim)
        if self.dim >= 5:
            lims.append(self.ulim)
        if self.dim >= 6:
            lims.append(self.vlim)
        return lims[: self.dim]

    def _max_radius(self):
        lims = self._coord_limits()
        return float(np.sqrt(sum(l[1] ** 2 for l in lims)))

    def _generate_hyperplanes_highdim(self):
        """与 dim==3 相同：随机单位法向、距离，并用 hbar 间距筛超平面。"""
        hbar = []
        max_distance = self._max_radius()
        vectors = np.zeros((self.hyperplane_num, self.dim))
        distances = np.zeros((self.hyperplane_num,))
        i = 0
        while i < self.hyperplane_num:
            distances[i] = 2.5 * max_distance * np.random.rand(1) / 5.0 + max_distance / 5.0
            v = np.random.randn(self.dim)
            v = v / (np.linalg.norm(v) + 1e-15)
            vectors[i, :] = v
            if i == 0:
                hbar.append(distances[i] * vectors[i, :])
                i += 1
            else:
                flag = True
                hbar_temp = distances[i] * vectors[i, :]
                for item in hbar:
                    if np.linalg.norm(hbar_temp - item) <= 10.0 * self.max_distance_from_hyperplane:
                        flag = False
                        break
                if flag:
                    hbar.append(hbar_temp)
                    i += 1
        self.ground_truth_A = vectors
        self.ground_truth_b = -distances
        return vectors, distances
        
    def get_data(self):
        vectors, distances = self.generate_hyperplanes_feature()
        data = np.zeros((1, self.dim))
        sum_distance = 0.
        for i in range(self.hyperplane_num):
            d, total_distance = self.get_points_hyperplane(vectors[i], distances[i])
            data = np.vstack((data, d))
            sum_distance += total_distance
        return data[1:,], sum_distance, vectors, distances
       
    
    
    def generate_hyperplanes_feature(self):
        if self.dim == 2:
            hbar = []
            max_distance = np.sqrt(self.xlim[1]**2 +self.ylim[1]**2)
            vectors = np.zeros((self.hyperplane_num, self.dim))
            distances = np.zeros((self.hyperplane_num,))
            i = 0
            while i < self.hyperplane_num:
                distances[i] = 2.5 * max_distance * np.random.rand(1) / 5. + max_distance / 5.
                theta = 2 * math.pi * np.random.rand(1) - math.pi
                vector = [math.cos(theta), math.sin(theta)]
                vectors[i, :] = np.reshape(np.array(vector), (1, self.dim))
                if i == 0:
                    hbar.append(distances[i] * vectors[i, :])
                    i += 1
                else:
                    flag = True
                    hbar_temp = distances[i] * vectors[i, :]
                    for item in hbar:
                        if np.linalg.norm(hbar_temp - item) <= 10. * self.max_distance_from_hyperplane:
                            flag = False
                            break
                    if flag:
                        hbar.append(hbar_temp)
                        i += 1
            self.ground_truth_A = vectors
            self.ground_truth_b = -distances
        elif self.dim == 3:
            hbar = []
            max_distance = np.sqrt(self.xlim[1]**2 +self.ylim[1]**2 +self.zlim[1]**2)
            vectors = np.zeros((self.hyperplane_num, self.dim))
            distances = np.zeros((self.hyperplane_num,))
            i = 0
            while i < self.hyperplane_num:
                distances[i] = 2.5 * max_distance * np.random.rand(1) / 5. + max_distance / 5.
                theta = 2 * math.pi * np.random.rand(1) - math.pi
                phi = math.pi * np.random.rand(1) - math.pi / 2.
                vector = [math.cos(phi) * math.cos(theta), math.cos(phi) * math.sin(theta), math.sin(phi)]
                vectors[i, :] = np.reshape(np.array(vector), (1, self.dim))
                if i == 0:
                    hbar.append(distances[i] * vectors[i, :])
                    i += 1
                else:
                    flag = True
                    hbar_temp = distances[i] * vectors[i, :]
                    for item in hbar:
                        if np.linalg.norm(hbar_temp - item) <= 10. * self.max_distance_from_hyperplane:
                            flag = False
                            break
                    if flag:
                        hbar.append(hbar_temp)
                        i += 1
            self.ground_truth_A = vectors
            self.ground_truth_b = -distances
        elif self.dim in (4, 5, 6):
            vectors, distances = self._generate_hyperplanes_highdim()
        else:
            raise ValueError(f"Random: unsupported dim={self.dim}, expected 2..6")
        return vectors, distances



    def get_points_hyperplane(self, vector, distance):
        if self.dim == 2:
            x_start = 0.         # the x of start point on line
            y_start = 0.         # the y of start point on line
            dirL = [0., 1.]      # unit vector along line
            disL_lim = [-np.inf, np.inf] # distance range on line     in xlim, ylim
            if vector[1] == 0.:
                x_start = distance / vector[0]
                y_start = self.ylim[0]
                dirL = [0., 1.]
                disL_lim = self.ylim
            elif vector[0] == 0.:
                x_start = self.xlim[0]
                y_start = distance / vector[1]
                dirL = [1., 0.]
                disL_lim = self.xlim
            else:
                x_start = self.xlim[0]
                y_start = (distance - vector[0] * x_start) / vector[1]
                temp = [1., -vector[0] / vector[1]]
                norm = np.linalg.norm(np.array(temp))
                dirL = [temp[0]/norm, temp[1]/norm]
                disL_lim = [0., 0.]
                if temp[1] > 0.:
                    disL_lim[0] = max((self.xlim[0] - x_start) / dirL[0], (self.ylim[0] - y_start) / dirL[1])
                    disL_lim[1] = min((self.xlim[1] - x_start) / dirL[0], (self.ylim[1] - y_start) / dirL[1])
                else:
                    disL_lim[0] = max((self.xlim[0] - x_start) / dirL[0], (self.ylim[1] - y_start) / dirL[1])
                    disL_lim[1] = min((self.xlim[1] - x_start) / dirL[0], (self.ylim[0] - y_start) / dirL[1])
            
            num_point = self.min_points_on_hyperplane + int((self.max_points_on_hyperplane - self.min_points_on_hyperplane) * np.random.rand(1))
            distance_list = np.sort(disL_lim[0] + (disL_lim[1] - disL_lim[0]) * np.random.rand(num_point))


            result = np.zeros((num_point, self.dim))
            total_distances = 0.
            for index, distance_along_hyerplane in enumerate(distance_list):
                hyperplane_distance = 2 * self.max_distance_from_hyperplane * np.random.rand(1) - self.max_distance_from_hyperplane
                temp1 = np.array([x_start + distance_along_hyerplane * dirL[0], y_start + distance_along_hyerplane * dirL[1]])
                temp2 = np.array([hyperplane_distance * vector[0], hyperplane_distance * vector[1]])
                
                if temp1[0] + temp2[0] < self.xlim[0]:
                    hyperplane_distance = (self.xlim[0] - temp1[0]) / vector[0]
                elif temp1[0] + temp2[0] > self.xlim[1]:
                    hyperplane_distance = (self.xlim[1] - temp1[0]) / vector[0]
                temp2 = np.array([hyperplane_distance * vector[0], hyperplane_distance * vector[1]])
                
                if temp1[1] + temp2[1] < self.ylim[0]:
                    hyperplane_distance = (self.ylim[0] - temp1[1]) / vector[1]
                elif temp1[1] + temp2[1] > self.ylim[1]:
                    hyperplane_distance = (self.ylim[1] - temp1[1]) / vector[1]
                temp2 = np.array([hyperplane_distance * vector[0], hyperplane_distance * vector[1]])
                
                result[index][0] = temp1[0] + temp2[0]
                result[index][1] = temp1[1] + temp2[1]
                
                total_distances += np.abs(hyperplane_distance)
                
            return result, total_distances
        
        elif self.dim == 3:
            point = distance * vector
            dx = dz = dy = 1.
            if vector[0] != 0.:
                dx = (- vector[1] * dy - vector[2] * dz) / vector[0]
            elif vector[1] != 0.:
                dy = (- vector[0] * dx - vector[2] * dz) / vector[1]
            else:
                dz = (- vector[0] * dx - vector[1] * dy) / vector[2]
            
            vector_on_hyperplane1 = np.array([dx, dy, dz])
            vector_on_hyperplane2 = np.cross(vector, vector_on_hyperplane1)
            norm1 = np.linalg.norm(vector_on_hyperplane1)
            norm2 = np.linalg.norm(vector_on_hyperplane2)
            
            vector_on_hyperplane1 = vector_on_hyperplane1 / norm1
            vector_on_hyperplane2 = vector_on_hyperplane2 / norm2
            
            num_point = self.min_points_on_hyperplane + int((self.max_points_on_hyperplane - self.min_points_on_hyperplane) * np.random.rand(1))
            result = np.zeros((num_point, self.dim))
            total_distances = 0.
            for i in range(num_point):
                distance1 = self.xlim[0] + (self.xlim[1] - self.xlim[0]) * np.random.rand(1)
                distance2 = self.ylim[0] + (self.ylim[1] - self.ylim[0]) * np.random.rand(1)
                distance3 = 2 * self.max_distance_from_hyperplane * np.random.rand(1) - self.max_distance_from_hyperplane
                
                result[i][0] = point[0] + distance1 * vector_on_hyperplane1[0] + distance2 * vector_on_hyperplane2[0] + distance3 * vector[0]
                result[i][1] = point[1] + distance1 * vector_on_hyperplane1[1] + distance2 * vector_on_hyperplane2[1] + distance3 * vector[1]
                result[i][2] = point[2] + distance1 * vector_on_hyperplane1[2] + distance2 * vector_on_hyperplane2[2] + distance3 * vector[2]
                
                total_distances += np.abs(distance3)
            
            return result, total_distances

        elif self.dim in (4, 5, 6):
            return self._get_points_hyperplane_highdim(vector, distance)

        else:
            raise ValueError(f"Random.get_points_hyperplane: unsupported dim={self.dim}")

    def _tangent_basis(self, n):
        """单位法向 n，返回 shape (D, D-1) 的列正交基，张成超平面切空间。"""
        D = self.dim
        n = np.asarray(n, dtype=np.float64).reshape(-1)
        n = n / (np.linalg.norm(n) + 1e-15)
        _, _, vh = np.linalg.svd(n.reshape(1, -1), full_matrices=True)
        return vh[1:, :].T

    def _get_points_hyperplane_highdim(self, vector, distance):
        """与 dim==3 类似：p0 = d·n，在切空间用各坐标轴范围均匀系数采样，沿法向加噪声。"""
        n = np.asarray(vector, dtype=np.float64)
        n = n / (np.linalg.norm(n) + 1e-15)
        p0 = distance * n
        basis = self._tangent_basis(n)
        lims = self._coord_limits()
        num_tangent = self.dim - 1
        num_point = self.min_points_on_hyperplane + int(
            (self.max_points_on_hyperplane - self.min_points_on_hyperplane) * np.random.rand(1)
        )
        result = np.zeros((num_point, self.dim))
        total_distances = 0.0
        for i in range(num_point):
            coeffs = np.zeros(num_tangent)
            for k in range(num_tangent):
                low, high = lims[k][0], lims[k][1]
                coeffs[k] = low + (high - low) * np.random.rand()
            noise = 2 * self.max_distance_from_hyperplane * np.random.rand(1) - self.max_distance_from_hyperplane
            noise = float(noise)
            result[i, :] = p0 + basis @ coeffs + noise * n
            total_distances += np.abs(noise)
        return result, total_distances





if __name__ == "__main__":
    generator = Random(2, 2, max_distance_from_hyperplane = 0.1, X_limit=[-5., 5.], Y_limit=[-5., 5.])
    data = generator.get_data()
    print(data)
    
    
    X = []
    Y = []
    for pos in data:
        X.append(pos[0])
        Y.append(pos[1])
        
    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(X, Y, color="black", s = 10)
    
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    # fig.savefig('ellipsoid.png')
    plt.show()