import numpy as np
import matplotlib.pyplot as plt
import datetime
import cvxpy as cp
import timeit
from utils import get_data, draw


# cvxpy 类似 Keras, 提供一种通用的编写方式，求解还是会调用其他求解器



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
        self.problem = None
        
        # Define variables
        self.hyperplanes_A = cp.Variable((self.num_hyperplanes, self.dim))  # Continuous variable for hyperplane normals
        self.hyperplanes_b = cp.Variable(self.num_hyperplanes)  # Continuous variable for hyperplane offsets
        self.e = cp.Variable(self.num_points, nonneg=True)  # Continuous non-negative slack variables
        self.z = cp.Variable((self.num_points, self.num_hyperplanes), boolean=True)  # Binary assignment variables
        self.distance_term = cp.Variable((self.num_points, self.num_hyperplanes), nonneg=True)  # Continuous non-negative distance terms
        
        self.sol_A = np.zeros((self.num_hyperplanes, self.dim))
        self.sol_b = np.zeros((self.num_hyperplanes, ))
    
        
    def set_initial_value(self):
        self.hyperplanes_A.value = self.ground_truth_A
        self.hyperplanes_b.value = self.ground_truth_b.flatten()

        
    def optimal_construction(self, setting_inital = True):
        if setting_inital:
            self.set_initial_value()
        
        # Define constraints
        constraints = []

        # Constraint: sum(z_ij) = 1 for all points i
        for i in range(self.num_points):
            constraints.append(cp.sum(self.z[i, :]) == 1)

        # Constraint: ||A[j]||_2 == 1 (approximated with bounds)
        # for j in range(self.num_hyperplanes):
        #    constraints.append(cp.norm(self.hyperplanes_A[j, :], 2) <= 1.0005)
        #    constraints.append(cp.norm(self.hyperplanes_A[j, :], 2) >= 0.9995)

        # Constraint: e_i >= distance_ij - M * (1 - z_ij)
        for i in range(self.num_points):
            for j in range(self.num_hyperplanes):
                # Compute distance expression
                distance_expr = self.hyperplanes_A[j, :] @ self.data[i, :] + self.hyperplanes_b[j]
                A_norm = cp.sum_squares(self.hyperplanes_A[j, :])
                
                # Define distance term
                constraints.append(self.distance_term[i, j] == cp.square(distance_expr) / A_norm)
                
                # Add slack variable constraint
                constraints.append(self.distance_term[i, j] <= self.e[i] + self.M * (1 - self.z[i, j]))

        # Objective function: Minimize sum of e_i
        objective = cp.Minimize(cp.sum(self.e))

        self.problem = cp.Problem(objective, constraints)
        
    
    
    def solve(self):
        # Define and solve the problem
        
        starttime = timeit.default_timer()
        # self.problem.solve(solver=cp.GUROBI)
        # self.problem.solve(solver='SCIP')
        # self.problem.solve(solver='SDPT3')
        self.problem.solve(solver=cp.SCS)
        t_diff = timeit.default_timer() - starttime
        

        self.sol_A = self.hyperplanes_A.value
        self.sol_b = self.hyperplanes_b.value
            # for key, value in self.z.items():
            #    print(self.z[key])
            
        return self.sol_A, self.sol_b, t_diff
        
    

if __name__ == "__main__":
    ground_truth_A, ground_truth_b, data = get_data(1, 2)
    
    A = np.ones((1, 2))
    b = np.ones((1, 1))
    solution = Multisource_Hyperplanes_Locations(1, ground_truth_A, ground_truth_b, data)
    try:
        solution.optimal_construction(setting_inital=True)
        A, b, time = solution.solve()
        print(A, b, time)
    except Exception as e:
        print(f"General error: {e}")
    print(ground_truth_A, ground_truth_b)
    draw(data, A, b, ground_truth_A, ground_truth_b)