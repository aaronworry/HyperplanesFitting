import numpy as np
import math
import time
from joblib import Parallel, delayed

K1 = 1.
K2 = 1.



class Point():
    def __init__(self, position, index):
        self.position = position
        self.hyperplane_id = -1.
        self.index = index

    def set_hyperplane_id(self, index):
        if self.hyperplane_id < -0.1:
            self.hyperplane_id = index
            

class Hyperplane():
    def __init__(self, normal=None, distance=None, point_index_list=[], score = np.inf):
        """
        initial a hyperplane
        :param distance: distance between the hyperplane and origin point [0., 0., 0.]
        :param normal: a normal vector point to the outside of hyperplane
        
        n^T * n = 1
        """
        self.distance = distance
        self.normal = normal
        
        self.point_index_list = point_index_list
        self.score = score
        
class Polyhedron():
    def __init__(self, dim, hps=[], epsilon=1e-10):
        """
        initial a polyhedron
        :param hps: a list of hyperplanes
        """
        self.dim = dim
        self.hps = hps
        self.epsilon = epsilon
        self.A = None
        self.b = None
        self.construct_Ab()
        
    def construct_Ab(self):
        num = len(self.hps)
        if num > 0:
            self.A = np.zeros((num, self.dim))
            self.b = np.zeros((num,))
            for i in range(len(self.hps)):
                self.A[i,:] = self.hps[i].normal
                self.b[i] = -self.hps[i].distance

    


class InitialSolution():
    def __init__(self, dim, data, parallel = True, min_distance_move_on_normal=0.2, max_left_points_num = 5, max_distance_delta = 0.2, max_hyperplanes_num = 4, min_points_num_hyperplane = 15, horizon_resolution = 3., vertical_resolution = 3.):
        self.dim = dim
        self.min_distance_move_on_normal = min_distance_move_on_normal
        self.max_left_points_num = max_left_points_num
        self.max_distance_delta = max_distance_delta
        self.max_hyperplanes_num = max_hyperplanes_num
        self.min_points_num_hyperplane = min_points_num_hyperplane
        self.horizon_resolution = horizon_resolution
        self.vertical_resolution = vertical_resolution
        self.data = data
        self.parallel = parallel
        
        self.points = []
        self.hyperplanes = []
        self.hyperplane_dict = {}
        self.get_point_from_data(data)
        
    def solve(self):
        normal_vector_list = self.sample_normal_vectors()
        self.find_initial_hyperplanes(normal_vector_list)
        return self.hyperplanes
        
    def get_point_from_data(self, data):
        for i in range(data.shape[0]):
            point = Point(data[i], i)
            self.points.append(point)
            

        
    def best_hyperplane_on_normal(self, normal, index_list, point_distance_list, point_index_list):
        # lower cost, but meet some strangy bugs. Sometimes can not work....
        if len(index_list) == 0:
            return None
        last = False
        points_index_list_one_hyperplane = []
        max_distance = point_distance_list[index_list[-1]]
        min_distance = point_distance_list[index_list[0]]

        flag_index_of_index_list = -1
        
        lower_bound = min_distance
        upper_bound = lower_bound + 2 * self.max_distance_delta
        for index in range(flag_index_of_index_list+1, len(index_list)):
            distance_id = index_list[index]
            if point_distance_list[distance_id] > upper_bound:
                break
            points_index_list_one_hyperplane.append(index_list[index])

        best_score = np.inf
        candidate_hyperplane = None
        while not last:        # need setting
            hyperplane = self.score_candidate_hyperplane(normal, points_index_list_one_hyperplane, point_distance_list, point_index_list)
            if hyperplane is not None and hyperplane.score < best_score:
                candidate_hyperplane = hyperplane
                best_score = hyperplane.score
            # update bound, and del element points_index_list_one_hyperplane
            lower_bound += self.min_distance_move_on_normal
            upper_bound = lower_bound + 2 * self.max_distance_delta

                
            for index in range(len(points_index_list_one_hyperplane)):
                if point_distance_list[points_index_list_one_hyperplane[index]] >= lower_bound:
                    points_index_list_one_hyperplane = points_index_list_one_hyperplane[index:]
                    lower_bound = point_distance_list[points_index_list_one_hyperplane[0]]
                    break
                flag_index_of_index_list += 1
            if point_distance_list[points_index_list_one_hyperplane[-1]] < lower_bound:
                points_index_list_one_hyperplane = []
                if flag_index_of_index_list + 1 < len(index_list):
                    lower_bound = point_distance_list[index_list[flag_index_of_index_list + 1]]
                else:
                    lower_bound = point_distance_list[-1]
                    last = True
            # update points_index_list_one_hyperplane
            upper_bound = lower_bound + 2 * self.max_distance_delta
            for index in range(flag_index_of_index_list+1, len(index_list)):
                if point_distance_list[index_list[index]] > upper_bound:
                    break
                points_index_list_one_hyperplane.append(index_list[index])
                if index == len(index_list) - 1:
                    last = True
            
            if last:
                hyperplane = self.score_candidate_hyperplane(normal, points_index_list_one_hyperplane, point_distance_list, point_index_list)
                if hyperplane is not None and hyperplane.score < best_score:
                    candidate_hyperplane = hyperplane
                    best_score = hyperplane.score

        return candidate_hyperplane
        
    
    def best_hyperplane_on_normal_1(self, normal, index_list, point_distance_list, point_index_list):
    
        if len(index_list) == 0:
            return None
    
        last = False
        points_index_list_one_hyperplane = []
        max_distance = point_distance_list[index_list[-1]]
        min_distance = point_distance_list[index_list[0]]
        
        
        lower_bound = min_distance
        upper_bound = lower_bound + 2 * self.max_distance_delta
        for index in range(len(index_list)):
            distance_id = index_list[index]
            if point_distance_list[distance_id] > upper_bound :
                break
            if point_distance_list[distance_id] >= lower_bound:
                points_index_list_one_hyperplane.append(index_list[index])
        best_score = np.inf
        candidate_hyperplane = None
        while not last:        # need setting
            hyperplane = self.score_candidate_hyperplane(normal, points_index_list_one_hyperplane, point_distance_list, point_index_list)
            if hyperplane is not None and hyperplane.score < best_score:
                candidate_hyperplane = hyperplane
                best_score = hyperplane.score
            # update bound, and del element points_index_list_one_hyperplane
            lower_bound += self.min_distance_move_on_normal
            upper_bound = lower_bound + 2 * self.max_distance_delta
            if upper_bound >= max_distance:
                last = True
                upper_bound = max_distance
                lower_bound = upper_bound - 2 * self.max_distance_delta
            points_index_list_one_hyperplane = []
            for index in range(len(index_list)):
                distance_id = index_list[index]
                if point_distance_list[distance_id] > upper_bound :
                    break
                if point_distance_list[distance_id] >= lower_bound:
                    points_index_list_one_hyperplane.append(index_list[index])
            
            if last:
                hyperplane = self.score_candidate_hyperplane(normal, points_index_list_one_hyperplane, point_distance_list, point_index_list)
                if hyperplane is not None and hyperplane.score < best_score:
                    candidate_hyperplane = hyperplane
                    best_score = hyperplane.score

        return candidate_hyperplane
        
        
        
    def score_candidate_hyperplane(self, normal, index_list, point_distance_list, point_index_list):
        if len(index_list) < self.min_points_num_hyperplane:
            return None
        
        # score = - K2 * (len(index_list) - self.min_points_num_hyperplane) / self.min_points_num_hyperplane
        
        distance_list = []
        point_list = []
        for index in index_list:
            distance_list.append(point_distance_list[index])
            point_list.append(point_index_list[index])
        distance = np.mean(distance_list)
        var = np.var(distance_list)
        
        # score += K1 * var
        
        score = -1 * float(len(index_list))
        
        
        return Hyperplane(normal, distance, point_list, score)

    def left_points(self):
        num = 0
        for point in self.points:
            if point.hyperplane_id < -0.5:
                num += 1
        return num

    def sample_normal_vectors(self, number=None):
        result = []
        if self.dim == 2:
            if number is None or number < 5.:
                theta_degree = -180.
                while theta_degree < 180.:
                    theta = theta_degree * np.pi / 180.
                    result.append(np.array([np.cos(theta), np.sin(theta)]))
                    theta_degree += self.horizon_resolution
            else:
                number = int(number)
                d_theta = 360 / number
                theta_degree = np.random.rand(1)[0] * 360. - 180.
                k = 0
                while k < number:
                    theta = theta_degree * np.pi / 180.
                    result.append(np.array([np.cos(theta), np.sin(theta)]))
                    theta_degree += d_theta
                    k += 1
        elif self.dim == 3:
            if number is None or number < 5.:
                result = [np.array(0., 0., 1.), np.array(0., 0., -1.)]
                theta_degree = -180.
                while theta_degree <= 180:
                    theta = theta_degree * np.pi / 180.
                    phi_degree = -88.
                    while phi_degree < 88.:
                        phi = phi_degree * np.pi / 180.
                        result.append(np.array([np.cos(theta)*np.cos(phi), np.sin(theta)*np.cos(phi), np.sin(phi)]))
                        phi_degree += self.vertical_resolution
                    theta_degree += self.horizon_resolution
            else:
                # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
                phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

                for i in range(number):
                    z = 1 - (i / float(number - 1)) * 2  # z goes from 1 to -1
                    radius = math.sqrt(1 - z * z)  # radius at z

                    theta = phi * i  # golden angle increment

                    x = math.cos(theta) * radius
                    y = math.sin(theta) * radius

                    result.append(np.array([x, y, z]))
                    
                """
                indices = np.arange(0, number, dtype=float) + 0.5
                phi = np.arccos(1 - 2*indices/number)
                theta = pi * (1 + 5**0.5) * indices
                result.append(np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]))
                """
                
        return result

    def find_initial_hyperplanes(self, normal_vector_list):
        while True:
            if len(self.hyperplanes) >= self.max_hyperplanes_num or self.left_points() < self.max_left_points_num:
                break
            
            if self.parallel:
                # why slower ?    Loading parallel modules?
                job_num = len(normal_vector_list)
                par_sol = Parallel(n_jobs=job_num, prefer=None)(delayed(self.get_hyperplane_on_normal)(vector, self.points) for vector in normal_vector_list)
                for i in range(job_num):
                    result = par_sol[i]
                    if result is not None:
                        self.hyperplane_dict[i] = result
            else:
                i = 0
                for normal_vector in normal_vector_list:
                    result = self.get_hyperplane_on_normal(normal_vector, self.points)
                    i += 1
                    if result is not None:
                        self.hyperplane_dict[i] = result
            
            hyperplane = self.best_hyperplane()
            
            
            if hyperplane is not None:
                self.hyperplanes.append(hyperplane)
                point_ids = hyperplane.point_index_list
                for point in self.points:
                    if point.index in point_ids:
                        point.set_hyperplane_id(len(self.hyperplanes) - 1)
                hyperplane = None
            else:
                break
                
    def get_hyperplane_on_normal(self, vector, points):
        point_distance_list = []
        point_id_list = []
        index_list = []

        for point in points:
            if point.hyperplane_id < -0.5:
                distance_temp = np.dot(point.position, vector)
                if distance_temp >= 0.:
                    point_distance_list.append(distance_temp)
                    point_id_list.append(point.index)

        if len(point_distance_list) > 0:
            index_list = np.argsort(np.array(point_distance_list))
        
        candidate_hyperplane = self.best_hyperplane_on_normal(vector, index_list, point_distance_list, point_id_list)
        return candidate_hyperplane
                
    def best_hyperplane(self):
        if len(self.hyperplane_dict) == 0:
            return None
        score = np.inf
        hyperplane = None
        for key, value in self.hyperplane_dict.items():
            temp_score = value.score
            if temp_score < score:
                hyperplane = value
                score = temp_score
        self.hyperplane_dict = {}
        return hyperplane
        
        
    
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.random_data import Random
    from view.plot import ResultViewer
    generator = Random(2, 4, max_distance_from_hyperplane = 0.1, min_points_on_hyperplane = 30, max_points_on_hyperplane = 30, X_limit=[-5., 5.], Y_limit=[-5., 5.])
    data, _ = generator.get_data()
    ground_truth_A, ground_truth_b = generator.ground_truth_A, generator.ground_truth_b
    ground_truth_hyperplanes = []
    for i in range(len(ground_truth_b)):
        ground_truth_hyperplanes.append(Hyperplane(ground_truth_A[i], -ground_truth_b[i]))
    ground_truth_poly = Polyhedron(2, ground_truth_hyperplanes)
    
    
    ALG = InitialSolution(dim=2, data=data, parallel = False, min_distance_move_on_normal=1., max_distance_delta=0.8, horizon_resolution = 10.)
    ALG.solve()
    for hp in ALG.hyperplanes:
        print(hp.normal, hp.distance)
    polyhedron = Polyhedron(2, ALG.hyperplanes)
    
    
    viewer = ResultViewer(dim = 2, data = data, ground_truth=ground_truth_poly, initial_hyperplanes = polyhedron, convex_region_hyperplanes = None, X_limit = [-5., 5.], Y_limit = [-5., 5.])
    viewer.draw_result()
    viewer.show(False, True)