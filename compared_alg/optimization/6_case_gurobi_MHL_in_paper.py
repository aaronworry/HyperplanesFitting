import numpy as np
import matplotlib.pyplot as plt
import datetime
import gurobipy as gp
from gurobipy import GRB
import timeit
from utils import get_data, draw

# CasADi
# cvxpy



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
        
        
        
        #origin
        
        
        #A 松弛法
        #添加约束 v_i + u_j >= e_i
        #obj sum(u) + sum(v)
        
        #B 目标函数 + 1/2 直线系数的平方和
        
        #C
        #添加二次约束，直线的系数平方和为1 ，可以松弛在一定范围内
        #self.model.addQConstr(x*x + y*y == 1)
        
        # 每一条直线上点太多时，也会无法求解出较优的值    陷入局部最优
        #实验1：原   : 会陷入局部最优 A = 0, b = 0: 此时满足条件，但是超平面有问题
        #实验2：原 + A:         同实验1
        #实验3： 原 + B         同实验1
        #实验4： 原 + C         同实验1    A = [0, 0.999]
        #实验5:  原 + A + B     同实验1
        #实验6： 原 + A + C     同实验4
        
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
    data = get_data(1, 2)
    A = np.ones((1, 2))
    b = np.ones((1, 1))
    solution = Multisource_Hyperplanes_Locations(1, data)
    try:
        solution.optimal_construction_A_C()
        A, b, time = solution.solve()
        print(A, b, time)
    except gp.GurobiError as e:
        print(f"Gurobi error: {e.errno} - {e}")
    except Exception as e:
        print(f"General error: {e}")
    draw(data, A, b)