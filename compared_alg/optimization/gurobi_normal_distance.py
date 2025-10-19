import numpy as np
import matplotlib.pyplot as plt
import datetime
import gurobipy as gp
from gurobipy import GRB, nlfunc
import timeit
from utils import get_data, draw


class Multisource_Hyperplanes_Locations():
    def __init__(self, num, GN, Gd, data, tol=1.e-9):
        self.tol = tol
        self.num_hyperplanes = num
        self.data = data
        self.num_points = data.shape[0]
        self.dim = data.shape[1]
        self.ground_truth_N = GN
        self.ground_truth_d = Gd
        self.M = 100
        self.model = gp.Model()
        self.hyperplanes_normal = self.model.addVars(self.num_hyperplanes, self.dim, vtype=GRB.CONTINUOUS, name="N")
        self.hyperplanes_distance = self.model.addVars(self.num_hyperplanes, vtype=GRB.CONTINUOUS, lb=0.0, name="d")
        # error_i^2
        self.e = self.model.addVars(self.num_points, vtype=GRB.CONTINUOUS, lb=0.0, name="e")
        # z_ij
        self.z = self.model.addVars(self.num_points, self.num_hyperplanes, vtype=GRB.BINARY, name="z")
        self.distance_term = self.model.addVars(self.num_points, self.num_hyperplanes, vtype=GRB.CONTINUOUS, lb=0.0, name="distance")
        
        
        self.sol_N = np.zeros((self.num_hyperplanes, self.dim))
        self.sol_p = np.zeros((self.num_hyperplanes, self.dim))
        self.sol_d = np.zeros((self.num_hyperplanes, 1))
    
        
    def set_initial_value(self):
        for key, value in self.hyperplanes_normal.items():
            self.hyperplanes_normal[key].Start = float(self.ground_truth_N[key])
        for key, value in self.hyperplanes_distance.items():
            self.hyperplanes_distance[key].Start = float(self.ground_truth_d[key])

        
    def optimal_construction(self):
        # sum z_ij = 1
        for i in range(self.num_points):
            self.model.addConstr(gp.quicksum(self.z[i,j] for j in range(self.num_hyperplanes)) == 1)
        
        
        # np.linalg.norm(normal[i]) == 1 
        for i in range(self.num_hyperplanes):
            self.model.addConstr(gp.quicksum(self.hyperplanes_normal[i, j]*self.hyperplanes_normal[i, j] for j in range(self.dim)) <= 1 + self.tol)
            self.model.addConstr(gp.quicksum(self.hyperplanes_normal[i, j]*self.hyperplanes_normal[i, j] for j in range(self.dim)) >= 1 - self.tol)
        
        
        # e_i >= distance_ij - M (1 - z_ij)
        
        # nX - d = 0
        for i in range(self.num_points):
            for j in range(self.num_hyperplanes):
                distance_expr = gp.quicksum(self.hyperplanes_normal[j, k] * self.data[i, k] for k in range(self.dim)) - self.hyperplanes_distance[j]
                self.model.addConstr(self.distance_term[i, j] <= self.e[i] + self.M * (1 - self.z[i, j]), name=f"pow_dis_{i}_{j}")
                self.model.addConstr(self.distance_term[i, j] == distance_expr * distance_expr)
                # self.model.addConstr(distance_expr <= self.e[i] + self.M * (1 - self.z[i, j]), name=f"abs_pos_{i}_{j}")
                # self.model.addConstr(-distance_expr <= self.e[i] + self.M * (1 - self.z[i, j]), name=f"abs_neg_{i}_{j}")


        obj = gp.quicksum(self.e[i] for i in range(self.num_points))
        self.model.setObjective(obj, GRB.MINIMIZE)
        
        # balance
        # self.model.setParam("MIPFocus", 0)
        # Set MIPFocus parameter to focus on finding feasible solutions quickly
        # self.model.setParam("MIPFocus", 1)
        # Set MIPFocus parameter to focus on proving optimality
        self.model.setParam("MIPFocus", 2)
        # Set MIPFocus parameter to focus on bound improvement
        # self.model.setParam("MIPFocus", 3)
    
    
    def solve(self):
        starttime = timeit.default_timer()
        self.model.optimize()
        t_diff = timeit.default_timer() - starttime
        
        if self.model.status == GRB.OPTIMAL:
            print(f"Optimal solution:")
            print(f"Objective value: {self.model.ObjVal}")
            
            for key, value in self.hyperplanes_normal.items():
                self.sol_N[key] = self.hyperplanes_normal[key].x
            for key, value in self.hyperplanes_distance.items():
                self.sol_d[key] = self.hyperplanes_distance[key].x
            # for key, value in self.z.items():
            #    print(self.z[key])
            
            return self.sol_N, self.sol_d, t_diff
        elif self.model.status == GRB.INFEASIBLE:
            print("Model is infeasible.")
        else:
            print(f"Optimization ended with status {model.status}")
        
        return None, None, t_diff
    
    
    
    
    



if __name__ == "__main__":
    ground_truth_A, ground_truth_b, normal, point, distance, data = get_data(1, 2, num_range = [4, 5])
    
    N = np.ones((1, 2))
    p = np.ones((1, 2))
    A = np.ones((1, 2))
    b = np.ones((1, 1))
    d = np.ones((1, 1))
    solution = Multisource_Hyperplanes_Locations(1, normal, distance, data)
    try:
        solution.optimal_construction()
        # solution.set_initial_value()
        N, d, time = solution.solve()
    except gp.GurobiError as e:
        print(f"Gurobi error: {e.errno} - {e}")
    except Exception as e:
        print(f"General error: {e}")
    print(ground_truth_A, ground_truth_b)
    A[0, 0] = N[0, 0]
    A[0, 1] = N[0, 1]
    b[0, 0] = -d[0, 0]      # Nx - d = 0
    print(A, b, time)
    draw(data, A, b, ground_truth_A, ground_truth_b)