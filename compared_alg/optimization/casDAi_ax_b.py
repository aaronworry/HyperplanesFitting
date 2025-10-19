import numpy as np
import matplotlib.pyplot as plt
import datetime
import casadi as ca
import timeit
from utils import get_data, draw

class Multisource_Hyperplanes_Locations():
    def __init__(self, num, GA, Gb, data, tol=1.e-5):
        self.tol = tol
        self.num_hyperplanes = num
        self.data = data
        self.num_points = data.shape[0]
        self.dim = data.shape[1]
        self.ground_truth_A = GA
        self.ground_truth_b = Gb
        self.M = 500
        self.C = 100
        self.model = ca.Opti()
        self.hyperplanes_A = self.model.variable(self.num_hyperplanes, self.dim)        # hyperplanes_A
        self.hyperplanes_b = self.model.variable(self.num_hyperplanes)             # hyperplanes_b
        self.e = self.model.variable(self.num_points)                  # error terms
        # z_ij
        self.z = self.model.integer(self.num_points, self.num_hyperplanes) # assignment variables (binary)
        # z = self.model.binary()
        self.distance_term = self.model.variable(self.num_points, self.num_hyperplanes)
        
        
        self.sol_A = np.zeros((self.num_hyperplanes, self.dim))
        self.sol_b = np.zeros((self.num_hyperplanes, 1))
        
    
        
    def set_initial_value(self):
        self.model.set_initial(self.hyperplanes_A, np.zeros((2, 3)))
        self.model.set_initial(self.hyperplanes_b, np.zeros((2, 1)))

        
    def optimal_construction(self):
        # Binary constraints on z
        self.model.subject_to(self.z >= 0)
        self.model.subject_to(self.z <= 1)

        # Error terms must be non-negative
        self.model.subject_to(self.e >= 0)
        self.model.subject_to(self.distance >= 0)

        # 1. Each point must be assigned to exactly one hyperplane
        for i in range(self.num_points):
            self.model.subject_to(ca.sums1(self.z[i, :]) == 1)

        # 2. Hyperplane normalization constraints (||A[i]||^2 ≈ 1)
        for j in range(self.num_hyperplanes):
            norm_squared = ca.sumsqr(self.hyperplanes_A[j, :])
            self.model.subject_to(norm_squared <= 1.0005)
            self.model.subject_to(norm_squared >= 0.9995)

        # 3. Distance constraints
        for i in range(self.num_points):
            for j in range(self.num_hyperplanes):
                dot_product = ca.dot(self.hyperplanes_A[j, :], data[i, :]) + b[j]
                norm_squared = ca.sumsqr(self.hyperplanes_A[j, :])
                dist_expr = (dot_product)**2 / norm_squared

                # Distance definition
                self.model.subject_to(self.distance[i, j] == dist_expr)

                # Conditional constraint using big-M
                self.model.subject_to(self.distance[i, j] <= self.e[i] + 100 * (1 - self.z[i, j]))

        # Objective (example): minimize total error
        opti.minimize(ca.sum(e))
        # Solver options
        p_opts = {"verbose": False}
        s_opts = {"ma57_automatic_scaling": "yes"}
        self.model.solver('bonmin', p_opts, s_opts)  # BONMIN supports mixed-integer

    
    
    def solve(self):
        starttime = timeit.default_timer()
        sol = self.model.solve()
        t_diff = timeit.default_timer() - starttime
        
        try:
            self.sol_A = sol.value(A)
            self.sol_b = sol.value(b)
            z_val = sol.value(z)
            e_val = sol.value(e)
            print("Optimization succeeded.")
            return self.sol_A, self.sol_b, t_diff
        except RuntimeError as e:
            print("Solver failed:", e)
        return None, None, t_diff
        



if __name__ == "__main__":
    ground_truth_A, ground_truth_b, data = get_data(1, 2)
    
    A = np.ones((1, 2))
    b = np.ones((1, 1))
    solution = Multisource_Hyperplanes_Locations(1, ground_truth_A, ground_truth_b, data)
    try:
        solution.optimal_construction()
        solution.set_initial_value()
        A, b, time = solution.solve()
        print(A, b, time)
    except gp.GurobiError as e:
        print(f"Gurobi error: {e.errno} - {e}")
    except Exception as e:
        print(f"General error: {e}")
    print(ground_truth_A, ground_truth_b)
    draw(data, A, b, ground_truth_A, ground_truth_b)