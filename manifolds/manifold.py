import numpy as np


class RiemannianSubmanifold():
    def __init__(self, dim):
        self.dim = dim
        
    def inner_product(self, point, t_vector_a, t_vector_b):
        pass
        
    def projection(self, point, vector):
        pass
        
    def norm(self, point, t_vector):
        pass
        
    def random_point(self):
        pass
        
    def random_tangent_vector(self, point):
        pass
        
    def zero_vector(self, point):
        pass
        
    def dist(self, point_a, point_b):
        pass
        
    def riemannian_gradient(self, point, euclidean_gradient):
        pass
        
    def riemannian_hessian(self, point, euclidean_gradient, euclidean_hessian, t_vector):
        pass
        
    def retraction(self, point, t_vector):
        pass
        
    def transport(self, point_a, point_b, t_vector_a):
        pass
        
    def to_tangent_space(self, point, vector):
        pass
        
    def weingarten(self, point, t_vector, n_vector):
        pass
        
       
        