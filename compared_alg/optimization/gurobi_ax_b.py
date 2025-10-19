import numpy as np
import matplotlib.pyplot as plt
import datetime
import gurobipy as gp
from gurobipy import GRB, nlfunc
import timeit
from utils import get_data, draw

# CasADi
# cvxpy



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
        self.model = gp.Model()
        self.hyperplanes_A = self.model.addVars(self.num_hyperplanes, self.dim, vtype=GRB.CONTINUOUS, name="A")
        self.hyperplanes_b = self.model.addVars(self.num_hyperplanes, vtype=GRB.CONTINUOUS, name="b")
        # self.hyperplanes_A.Start    setting inital value before optimal
        # error_i^2
        self.e = self.model.addVars(self.num_points, vtype=GRB.CONTINUOUS, lb=0.0, name="e")
        # z_ij
        self.z = self.model.addVars(self.num_points, self.num_hyperplanes, vtype=GRB.BINARY, name="z")
        self.distance_term = self.model.addVars(self.num_points, self.num_hyperplanes, vtype=GRB.CONTINUOUS, lb=0.0, name="distance")
        
        
        self.sol_A = np.zeros((self.num_hyperplanes, self.dim))
        self.sol_b = np.zeros((self.num_hyperplanes, 1))
    
        
    def set_initial_value(self):
        for key, value in self.hyperplanes_A.items():
            self.hyperplanes_A[key].Start = float(self.ground_truth_A[key])
        for key, value in self.hyperplanes_b.items():
            self.hyperplanes_b[key].Start = float(self.ground_truth_b[key])

        
    def optimal_construction(self):
        # sum z_ij = 1
        for i in range(self.num_points):
            self.model.addConstr(gp.quicksum(self.z[i,j] for j in range(self.num_hyperplanes)) == 1)
        
        
        # np.linalg.norm(A[i]) == 1 
        for i in range(self.num_hyperplanes):
            self.model.addConstr(gp.quicksum(self.hyperplanes_A[i, j]*self.hyperplanes_A[i, j] for j in range(self.dim)) <= 1.0005)
            self.model.addConstr(gp.quicksum(self.hyperplanes_A[i, j]*self.hyperplanes_A[i, j] for j in range(self.dim)) >= 0.9995)
        
        
        # e_i >= distance_ij - M (1 - z_ij)
        # d <= e_i + M (1 - z_ij)
        for i in range(self.num_points):
            for j in range(self.num_hyperplanes):
                distance_expr = gp.quicksum(self.hyperplanes_A[j, k] * self.data[i, k] for k in range(self.dim)) + self.hyperplanes_b[j]
                A_norm = gp.quicksum(self.hyperplanes_A[j, k]*self.hyperplanes_A[j, k] for k in range(self.dim))
                self.model.addConstr(self.distance_term[i, j] == distance_expr * distance_expr / A_norm)
                self.model.addConstr(self.distance_term[i, j] <= self.e[i] + self.M * (1 - self.z[i, j]), name=f"abs_pos_{i}_{j}")
                # self.model.addConstr(-distance_expr * A_norm_inv <= self.e[i] + self.M * (1 - self.z[i, j]), name=f"abs_neg_{i}_{j}")
        

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
            
            for key, value in self.hyperplanes_A.items():
                self.sol_A[key] = self.hyperplanes_A[key].x
            for key, value in self.hyperplanes_b.items():
                self.sol_b[key] = self.hyperplanes_b[key].x
            # for key, value in self.z.items():
            #    print(self.z[key])
            
            return self.sol_A, self.sol_b, t_diff
        elif self.model.status == GRB.INFEASIBLE:
            print("Model is infeasible.")
        else:
            print(f"Optimization ended with status {model.status}")
        
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