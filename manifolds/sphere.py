from manifolds.manifold import RiemannianSubmanifold
import numpy as np



class SphereManifold(RiemannianSubmanifold):
    def __init__(self, dim):
        super().__init__(dim)
        pass
        
    def inner_product(self, point, t_vector_a, t_vector_b):
        return np.dot(t_vector_a, t_vector_b)
        
    def projection(self, point, vector):
        return vector - self.inner_product(point, point, vector) * point
        
    def norm(self, point, t_vector):
        return np.linalg.norm(t_vector)
        
    def random_point(self):
        point = np.random.normal(size=self.dim)
        return self._normalize(point)
        
    def random_tangent_vector(self, point):
        vector = np.random.normal(size=self.dim)
        return self._normalize(self.projection(point, vector))
        
    def zero_vector(self, point):
        return np.zeros(self.dim)
        
    def dist(self, point_a, point_b):
        inner = max(min(self.inner_product(point_a, point_a, point_b), 1), -1)
        return np.arccos(inner)
        
    def riemannian_gradient(self, point, euclidean_gradient):
        return self.projection(point, euclidean_gradient)
        
    def riemannian_hessian(self, point, euclidean_gradient, euclidean_hessian, t_vector):
        n_gradient = euclidean_gradient - self.projection(point, euclidean_gradient)
        return self.projection(point, euclidean_hessian) + self.weingarten(point, t_vector, n_gradient)
        
    def retraction(self, point, t_vector):
        return self._normalize(point + t_vector)
        
    def transport(self, point_a, point_b, t_vector_a):
        return self.projection(point_b, t_vector_a)
        
    def to_tangent_space(self, point, vector):
        return self.projection(point, vector)
        
    def weingarten(self, point, t_vector, n_vector):
        return (-self.inner_product(point, point, n_vector) * t_vector)
        
    def _normalize(self, vector):
        return vector / np.linalg.norm(vector)