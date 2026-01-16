import numpy as np
import matplotlib.pyplot as plt
import datetime
import gurobipy as gp
from gurobipy import GRB
import timeit
from utils import get_data, draw
# import sys
# sys.path.append("../..")
# from view.plot import ResultViewer
# from algorithm.initial_value import Hyperplane, Polyhedron




class Multisource_Hyperplanes_Locations():
    def __init__(self, num, data, tol=1.e-5):
        self.tol = tol
        self.num_hyperplanes = num
        self.data = data
        self.num_points = data.shape[0]
        self.dim = data.shape[1]
        self.M = 500
        self.C = 100
        self.model = gp.Model()
        self.hyperplanes_A = self.model.addVars(self.num_hyperplanes, self.dim, vtype=GRB.CONTINUOUS, name="A")
        self.hyperplanes_b = self.model.addVars(self.num_hyperplanes, vtype=GRB.CONTINUOUS, name="b")
        # self.hyperplanes_A.Start    setting inital value before optimal
        # error_i
        self.e = self.model.addVars(self.num_points, vtype=GRB.CONTINUOUS, lb=0.0, name="e")
        # z_ij
        self.z = self.model.addVars(self.num_points, self.num_hyperplanes, vtype=GRB.BINARY, name="z")
        
        self.sol_A = np.zeros((self.num_hyperplanes, self.dim))
        self.sol_b = np.zeros((self.num_hyperplanes, 1))
    
        
        
        
        
    def optimal_construction_origin(self):
               
        # sum z_ij = 1
        for i in range(self.num_points):
            self.model.addConstr(gp.quicksum(self.z[i,j] for j in range(self.num_hyperplanes)) == 1)
        
        # e_i >= distance_ij - M (1 - z_ij)
        # d <= e_i + M (1 - z_ij)
        for i in range(self.num_points):
            for j in range(self.num_hyperplanes):
                distance_expr = gp.quicksum(self.hyperplanes_A[j, k] * self.data[i, k] for k in range(self.dim)) + self.hyperplanes_b[j]
                self.model.addConstr(distance_expr <= self.e[i] + self.M * (1 - self.z[i, j]), name=f"abs_pos_{i}_{j}")
                self.model.addConstr(-distance_expr <= self.e[i] + self.M * (1 - self.z[i, j]), name=f"abs_neg_{i}_{j}")
        

        obj = gp.quicksum(self.e[i] for i in range(self.num_points))
        self.model.setObjective(obj, GRB.MINIMIZE)
        
        
        
        
    def optimal_construction_A(self):
        # u_i
        u = self.model.addVars(self.num_points, vtype=GRB.CONTINUOUS, name="u")
        # v_i
        v = self.model.addVars(self.num_points, vtype=GRB.CONTINUOUS, name="v")
        
        # sum z_ij = 1
        for i in range(self.num_points):
            self.model.addConstr(gp.quicksum(self.z[i,j] for j in range(self.num_hyperplanes)) == 1)
        
        # v_i + u_j >= e_i
        for i in range(self.num_points):
            for j in range(self.num_points):
                self.model.addConstr(v[i] + u[j] >= self.e[i])
        
        # e_i >= distance_ij - M (1 - z_ij)
        # d <= e_i + M (1 - z_ij)
        for i in range(self.num_points):
            for j in range(self.num_hyperplanes):
                distance_expr = gp.quicksum(self.hyperplanes_A[j, k] * self.data[i, k] for k in range(self.dim)) + self.hyperplanes_b[j]
                self.model.addConstr(distance_expr <= self.e[i] + self.M * (1 - self.z[i, j]), name=f"abs_pos_{i}_{j}")
                self.model.addConstr(-distance_expr <= self.e[i] + self.M * (1 - self.z[i, j]), name=f"abs_neg_{i}_{j}")
        
        

        obj = gp.quicksum(u[i]+v[i] for i in range(self.num_points))
        self.model.setObjective(obj, GRB.MINIMIZE)
    
    
    
    def optimal_construction_B(self):
        
        # sum z_ij = 1
        for i in range(self.num_points):
            self.model.addConstr(gp.quicksum(self.z[i,j] for j in range(self.num_hyperplanes)) == 1)
        
        # e_i >= distance_ij - M (1 - z_ij)
        # d <= e_i + M (1 - z_ij)
        for i in range(self.num_points):
            for j in range(self.num_hyperplanes):
                distance_expr = gp.quicksum(self.hyperplanes_A[j, k] * self.data[i, k] for k in range(self.dim)) + self.hyperplanes_b[j]
                self.model.addConstr(distance_expr <= self.e[i] + self.M * (1 - self.z[i, j]), name=f"abs_pos_{i}_{j}")
                self.model.addConstr(-distance_expr <= self.e[i] + self.M * (1 - self.z[i, j]), name=f"abs_neg_{i}_{j}")
        

        obj1 = gp.quicksum(self.C * self.e[i] for i in range(self.num_points))
        obj2 = gp.quicksum(self.hyperplanes_A[i, j]*self.hyperplanes_A[i, j] for i, j in np.ndindex(self.sol_A.shape))
        
        self.model.setObjective(obj1 + 0.5 * obj2, GRB.MINIMIZE)

        
    def optimal_construction_C(self):
        # sum z_ij = 1
        for i in range(self.num_points):
            self.model.addConstr(gp.quicksum(self.z[i,j] for j in range(self.num_hyperplanes)) == 1)
        
        # np.linalg.norm(A[i]) == 1 
        for i in range(self.num_hyperplanes):
            self.model.addConstr(gp.quicksum(self.hyperplanes_A[i, j]*self.hyperplanes_A[i, j] for j in range(self.dim)) <= 1 + self.tol)
            self.model.addConstr(gp.quicksum(self.hyperplanes_A[i, j]*self.hyperplanes_A[i, j] for j in range(self.dim)) >= 1 - self.tol)
        
        # e_i >= distance_ij - M (1 - z_ij)
        # d <= e_i + M (1 - z_ij)
        for i in range(self.num_points):
            for j in range(self.num_hyperplanes):
                distance_expr = gp.quicksum(self.hyperplanes_A[j, k] * self.data[i, k] for k in range(self.dim)) + self.hyperplanes_b[j]
                self.model.addConstr(distance_expr <= self.e[i] + self.M * (1 - self.z[i, j]), name=f"abs_pos_{i}_{j}")
                self.model.addConstr(-distance_expr <= self.e[i] + self.M * (1 - self.z[i, j]), name=f"abs_neg_{i}_{j}")
        

        obj = gp.quicksum(self.e[i] for i in range(self.num_points))
        self.model.setObjective(obj, GRB.MINIMIZE)
        
    def optimal_construction_A_B(self):
        # u_i
        u = self.model.addVars(self.num_points, vtype=GRB.CONTINUOUS, name="u")
        # v_i
        v = self.model.addVars(self.num_points, vtype=GRB.CONTINUOUS, name="v")
        
        # sum z_ij = 1
        for i in range(self.num_points):
            self.model.addConstr(gp.quicksum(self.z[i,j] for j in range(self.num_hyperplanes)) == 1)
        
        # v_i + u_j >= e_i
        for i in range(self.num_points):
            for j in range(self.num_points):
                self.model.addConstr(v[i] + u[j] >= self.e[i])
        
        # e_i >= distance_ij - M (1 - z_ij)
        # d <= e_i + M (1 - z_ij)
        for i in range(self.num_points):
            for j in range(self.num_hyperplanes):
                distance_expr = gp.quicksum(self.hyperplanes_A[j, k] * self.data[i, k] for k in range(self.dim)) + self.hyperplanes_b[j]
                self.model.addConstr(distance_expr <= self.e[i] + self.M * (1 - self.z[i, j]), name=f"abs_pos_{i}_{j}")
                self.model.addConstr(-distance_expr <= self.e[i] + self.M * (1 - self.z[i, j]), name=f"abs_neg_{i}_{j}")

        obj1 = gp.quicksum(self.C*u[i]+self.C*v[i] for i in range(self.num_points))
        obj2 = gp.quicksum(self.hyperplanes_A[i, j]*self.hyperplanes_A[i, j] for i, j in np.ndindex(self.sol_A.shape))
        
        self.model.setObjective(obj1 + 0.5 * obj2, GRB.MINIMIZE)
        
        
    def optimal_construction_A_C(self):
        # u_i
        u = self.model.addVars(self.num_points, vtype=GRB.CONTINUOUS, name="u")
        # v_i
        v = self.model.addVars(self.num_points, vtype=GRB.CONTINUOUS, name="v")
        
        # sum z_ij = 1
        for i in range(self.num_points):
            self.model.addConstr(gp.quicksum(self.z[i,j] for j in range(self.num_hyperplanes)) == 1)
        
        # v_i + u_j >= e_i
        for i in range(self.num_points):
            for j in range(self.num_points):
                self.model.addConstr(v[i] + u[j] >= self.e[i])
        
        # e_i >= distance_ij - M (1 - z_ij)
        # d <= e_i + M (1 - z_ij)
        for i in range(self.num_points):
            for j in range(self.num_hyperplanes):
                distance_expr = gp.quicksum(self.hyperplanes_A[j, k] * self.data[i, k] for k in range(self.dim)) + self.hyperplanes_b[j]
                self.model.addConstr(distance_expr <= self.e[i] + self.M * (1 - self.z[i, j]), name=f"abs_pos_{i}_{j}")
                self.model.addConstr(-distance_expr <= self.e[i] + self.M * (1 - self.z[i, j]), name=f"abs_neg_{i}_{j}")

        # np.linalg.norm(A[i]) == 1 
        for i in range(self.num_hyperplanes):
            self.model.addConstr(gp.quicksum(self.hyperplanes_A[i, j]*self.hyperplanes_A[i, j] for j in range(self.dim)) <= 1 + self.tol)
            self.model.addConstr(gp.quicksum(self.hyperplanes_A[i, j]*self.hyperplanes_A[i, j] for j in range(self.dim)) >= 1 - self.tol)
        
        
        obj = gp.quicksum(u[i]+v[i] for i in range(self.num_points))
        self.model.setObjective(obj, GRB.MINIMIZE)
        
        
    
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
            for key, value in self.z.items():
                print(self.z[key])
            
            return self.sol_A, self.sol_b, t_diff
        elif self.model.status == GRB.INFEASIBLE:
            print("Model is infeasible.")
        else:
            print(f"Optimization ended with status {model.status}")
        
        return None, None, t_diff
    
    
    
    
    



if __name__ == "__main__":
    gA, gb, vectors, points, distance, data = get_data(2, 2)
    A = np.ones((2, 2))
    b = np.ones((2, 1))
    solution = Multisource_Hyperplanes_Locations(2, data)
    try:
        solution.optimal_construction_origin()   # change methods
        A, b, time = solution.solve()
        print(A, b, time)
    except gp.GurobiError as e:
        print(f"Gurobi error: {e.errno} - {e}")
    except Exception as e:
        print(f"General error: {e}")
    draw(data, A, b, gA, gb)
    
    """
    ground_truth_hyperplanes = []
    for i in range(len(gb)):
        ground_truth_hyperplanes.append(Hyperplane(gA[i], -gb[i]))
    ground_truth_poly = Polyhedron(2, ground_truth_hyperplanes)
    
    r_hyperplanes = []
    for i in range(len(b)):
        r_hyperplanes.append(Hyperplane(A[i], -b[i]))
    polyhedron = Polyhedron(2, r_hyperplanes)

    print(data, A, b, gA, gb)

    viewer = ResultViewer(dim = 2, data = data, ground_truth=ground_truth_poly, initial_hyperplanes = polyhedron, convex_region_hyperplanes = None, X_limit = [-5., 5.], Y_limit = [-5., 5.])
    viewer.draw_result()
    viewer.show(False, True)
    """