import sys
sys.path.append("..")
from manifolds.sphere import SphereManifold
from solver.steepest_descent import SteepestDescent
from data.random_data import Random
from algorithm.initial_value import Hyperplane, InitialSolution, Polyhedron
from algorithm.manifold_optimization import ManifoldOptimization
from view.plot import ResultViewer
from joblib import Parallel, delayed
import numpy as np
import time



class HyperplanesFitting():
    def __init__(self, dim, data, parallel = False, method = "1", whether_initial_value = True):
        self.dim = dim
        self.data = data
        self.parallel = parallel
        self.hyperplanes = []
        self.method = method
        self.whether_initial_value = whether_initial_value
        
    def set_data(self, data):
        self.data = data
        
    def get_initial_value(self, parallel = False, delta=0.2, min_points_num_hyperplane = 20, horizon_resolution = 2.):
        solution = InitialSolution(dim=self.dim, data=self.data, parallel = parallel, delta = delta, min_points_num_hyperplane = min_points_num_hyperplane, horizon_resolution = horizon_resolution)
        start_time = time.time()
        solution.solve()
        dtime = time.time() - start_time
        self.hyperplanes = solution.hyperplanes
        # a list of Hyperplane
        
    def random_initial_value(self, hyperplane_true_num, parallel = False, delta=0.2, min_points_num_hyperplane = 20, horizon_resolution = 2.):
        solution = InitialSolution(dim=self.dim, data=self.data, parallel = parallel, delta=delta, min_points_num_hyperplane = min_points_num_hyperplane, horizon_resolution = horizon_resolution)
        normal_vector_list = solution.sample_normal_vectors(hyperplane_true_num)
        hps = []
        for item in normal_vector_list:
            hp = Hyperplane(item, 2.5)
            hps.append(hp)
        self.hyperplanes = hps
        
    def solve(self, true_num):
        if self.whether_initial_value:
            self.get_initial_value()
            if self.method == "1":
                self.method_one()
            elif self.method == "2":
                self.method_two()
            else:
                self.method_one()
                self.method_two()
            return self.hyperplanes
        else:
            Hyperplanes = []
            """
            # cost much
            for k in range(2, 7):
                self.random_initial_value(k)
                if self.method == "1":
                    self.method_one()
                elif self.method == "2":
                    self.method_two()
                else:
                    self.method_one()
                    self.method_two()
                if k == true_num:
                    Hyperplanes = self.hyperplanes
            """
            self.random_initial_value(true_num)
            if self.method == "1":
                self.method_one()
            elif self.method == "2":
                self.method_two()
            else:
                self.method_one()
                self.method_two()
            Hyperplanes = self.hyperplanes
            return Hyperplanes
                    
            
        
        
        
    def cal_weights(self):
        # line 6 of algorithm 1
        n, m = len(self.data), len(self.hyperplanes)
        result = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                distance = abs(np.dot(self.data[i], self.hyperplanes[j].normal) - self.hyperplanes[j].distance)
                # result[i][j] = np.exp(-1. * distance)
                result[i][j] = 1. / distance**2          # best performance
                # result[i][j] = 1. / distance
        weight = result / result.sum(axis = 1).reshape(-1, 1)
        return weight
        
    def update_hyperplanes_weights(self, weight):
        if self.parallel:
            job_num = len(self.hyperplanes)
            par_sol = Parallel(n_jobs=job_num, prefer=None)(delayed(self.update_one_hyperplanes)(i, self.data, weight[:, i]) for i in range(job_num))
            for i in range(job_num):
                normal, distance = par_sol[i]
                self.hyperplanes[i].normal = normal
                self.hyperplanes[i].distance = distance
        else:
            for i in range(len(self.hyperplanes)):
                normal, distance = self.update_one_hyperplanes(i, self.data, weight[:, i])
                self.hyperplanes[i].normal = normal
                self.hyperplanes[i].distance = distance
            
        
    def update_one_hyperplanes(self, i, data, weight = None):
        # line 8,9 of algo. 1
        optimization = ManifoldOptimization(self.dim, data, manifold = SphereManifold(self.dim), initialvalue = self.hyperplanes[i], weights = weight)
        normal = optimization.solve(SteepestDescent())
        num_data = len(data)
        if weight is None:
            weight = np.array([1.] * num_data)
        distance = np.sum([weight[j] * np.dot(normal, data[j]) for j in range(num_data)]) / np.sum(weight)
        return normal, distance
        
    def method_one(self, threshold = 1e-2):
        n, m = len(self.data), len(self.hyperplanes)
        last_weight = np.zeros((n, m))
        iteration_num = 0
        while iteration_num <= 20:
            weight = self.cal_weights()
            self.update_hyperplanes_weights(weight)
            d_weight = weight - last_weight
            iteration_num += 1
            if np.max(d_weight) <= threshold:
                break
            last_weight = weight
        print("=========== iter_num=", iteration_num)
        
        # arange points to hyperplanes
        for hp in self.hyperplanes:
            hp.point_index_list = []
        for point_index in range(len(self.data)):
            distance_list = [abs(np.dot(self.data[point_index], self.hyperplanes[j].normal) - self.hyperplanes[j].distance) for j in range(len(self.hyperplanes))]
            hyperplane_ids = np.argsort(distance_list)[0]
            self.hyperplanes[hyperplane_ids].point_index_list.append(point_index)
        
    def method_two(self):
        if not self.whether_initial_value:
            for hp in self.hyperplanes:
                hp.point_index_list = []
            for point_index in range(len(self.data)):
                distance_list = [abs(np.dot(self.data[point_index], self.hyperplanes[j].normal) - self.hyperplanes[j].distance) for j in range(len(self.hyperplanes))]
                hyperplane_ids = np.argsort(distance_list)[0]
                self.hyperplanes[hyperplane_ids].point_index_list.append(point_index)
        
        if self.parallel:
            job_num = len(self.hyperplanes)
            par_sol = Parallel(n_jobs=job_num, prefer=None)(delayed(self.update_one_hyperplanes)(i, self.data[self.hyperplanes[i].point_index_list, :]) for i in range(job_num))
            for i in range(job_num):
                normal, distance = par_sol[i]
                self.hyperplanes[i].normal = normal
                self.hyperplanes[i].distance = distance
        else:
            for i in range(len(self.hyperplanes)):
                normal, distance = self.update_one_hyperplanes(i, self.data[self.hyperplanes[i].point_index_list, :])
                self.hyperplanes[i].normal = normal
                self.hyperplanes[i].distance = distance


if __name__ == "__main__":
    generator = Random(2, 4, max_distance_from_hyperplane = 0.1, min_points_on_hyperplane = 30, max_points_on_hyperplane = 30, X_limit=[-5., 5.], Y_limit=[-5., 5.])
    data, _, _, _ = generator.get_data()
    ground_truth_A, ground_truth_b = generator.ground_truth_A, generator.ground_truth_b
    ground_truth_hyperplanes = []
    for i in range(len(ground_truth_b)):
        ground_truth_hyperplanes.append(Hyperplane(ground_truth_A[i], -ground_truth_b[i]))
    ground_truth_poly = Polyhedron(2, ground_truth_hyperplanes)
    
    
    ALG = HyperplanesFitting(2, data, parallel = False, method = "3")
    ALG.solve()
    for hp in ALG.hyperplanes:
        print(hp.normal, hp.distance)
    polyhedron = Polyhedron(2, ALG.hyperplanes)
    
    
    viewer = ResultViewer(dim = 2, data = data, ground_truth=ground_truth_poly, initial_hyperplanes = polyhedron, convex_region_hyperplanes = None, X_limit = [-5., 5.], Y_limit = [-5., 5.])
    viewer.draw_result()
    viewer.show(True, True)
