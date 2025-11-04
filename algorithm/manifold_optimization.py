import numpy as np
import time

class ManifoldOptimization():
    def __init__(self, dim, data, manifold, initialvalue = None, weights = None):
        """
        initialvalue: a Hyperplane
        """
        self.dim = dim
        self.data = data
        self.num = len(self.data)
        self.manifold = manifold
        self.initialvalue = initialvalue
        self.weights = weights
        if self.weights is None:
            self.weights = np.array([1.] * self.num)

        
    def solve(self, method):
        solver = method
        if self.initialvalue is None:
            x, cost, iteration = solver.run(self.manifold, self.objective, self.riemannian_gradient)
            return x
        else:
            x, cost, iteration = solver.run(self.manifold, self.objective, self.riemannian_gradient, self.initialvalue.normal)
            return x
    
    def objective(self, x):
        mean = np.sum([self.weights[i] * x.dot(self.data[i]) for i in range(self.num)]) / np.sum(self.weights)
        return np.sum([self.weights[i] * (x.dot(self.data[i]) - mean)**2 for i in range(self.num)]) / np.sum(self.weights)
        
        
    def euclidean_gradient(self, x):
        # eq. 15, 16
        mean_data = np.array([0.] * self.dim)
        mean = 0.
        for i in range(self.num):
            mean += self.weights[i] * np.dot(x, self.data[i])
            mean_data += self.weights[i] * self.data[i]
        mean /= np.sum(self.weights)
        mean_data /= np.sum(self.weights)
        
        gradient = np.array([0.] * self.dim)
        for i in range(self.num):
            gradient += 2 * self.weights[i] * (x.dot(self.data[i]) - mean) * (self.data[i] - mean_data)
        gradient /= np.sum(self.weights)
        
        return gradient
    
    
    def riemannian_gradient(self, x):
        e_gradient = self.euclidean_gradient(x)
        return self.manifold.riemannian_gradient(x, e_gradient)


if __name__ == "__main__":
    
    import sys
    sys.path.append("..")
    import numpy as np
    from manifolds.sphere import SphereManifold
    from solver.steepest_descent import SteepestDescent
    
    data = np.array([[1., 1.], [2., 3.], [3., 5.], [4., 7.]])
    # Manifold = SphereManifold(2)
    optimization = ManifoldOptimization(2, data, manifold = SphereManifold(2))
    print(optimization.solve(SteepestDescent()))
