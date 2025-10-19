from solver.baseSolver import BaseSolver
from solver.lineSearch import BackTrackingLineSearcher
from copy import deepcopy
import time

class SteepestDescent(BaseSolver):
    def __init__(self, line_searcher=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if line_searcher is None:
            self._line_searcher = BackTrackingLineSearcher()
        else:
            self._line_searcher = line_searcher
        self.line_searcher = None
        
    def run(self, manifold, objective, riemannian_gradient, initial_point=None, reuse_line_searcher=False):
        if not reuse_line_searcher or self.line_searcher is None:
            self.line_searcher = deepcopy(self._line_searcher)
        line_searcher = self.line_searcher
        
        #initial 
        if initial_point is None:
            x = manifold.random_point()
        else:
            x = initial_point
            
        # Initialize iteration counter and timer
        iteration = 0
        start_time = time.time()

        while True:
            iteration += 1

            # Calculate new cost, grad and gradient_norm
            cost = objective(x)
            grad = riemannian_gradient(x)
            gradient_norm = manifold.norm(x, grad)

            print(iteration, cost, gradient_norm)


            # Descent direction is minus the gradient
            desc_dir = -grad

            # Perform line-search
            step_size, x = line_searcher.search(objective, manifold, x, desc_dir, cost, -(gradient_norm**2))

            stopping_criterion = self._check_stopping_criterion(
                start_time=start_time,
                step_size=step_size,
                gradient_norm=gradient_norm,
                iteration=iteration,
            )

            if stopping_criterion:
                if self._verbosity >= 1:
                    print(stopping_criterion)
                    print("")
                break

        return x, objective(x), iteration
        # return x, objective(x), iteration, stopping_criterion, step_size, gradient_norm
        
        
        
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    import numpy as np
    from manifolds.sphere import SphereManifold
    
    data = np.array([[1., 1.], [2., 3.], [3., 5.], [4., 7.]])
    
    def objective(x):
        mean = np.mean([x.dot(item) for item in data])
        return np.sum([(x.dot(item) - mean)**2 for item in data]) / float(len(data))
        
    def euclidean_gradient(x):
        mean_data = np.array([0., 0.])
        mean = 0.
        num = float(len(data))
        for item in data:
            mean += np.dot(x, item)
            mean_data += item
        mean /= num
        mean_data /= num
        
        gradient = np.array([0.] * 2)
        for item in data:
            gradient += 2 * (x.dot(item) - mean) * (item - mean_data)
        gradient /= num
        
        return gradient
    
        

    sphere = SphereManifold(2)          # x^T x = 1
    
    def riemannian_gradient(x):
        e_gradient = euclidean_gradient(x)
        return sphere.riemannian_gradient(x, e_gradient)
    
    
    Solver = SteepestDescent()
    x, cost, iteration = Solver.run(sphere, objective, riemannian_gradient)
    print(x, cost, iteration)