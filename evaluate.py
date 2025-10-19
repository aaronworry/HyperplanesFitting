# 1. average_distance
# 2. total cost function
# 3. 和ground truth 的关联                 mxn内积矩阵(m >= n)  可以交换列，使对角线上之和最小


import numpy as np

def cal_hbar(dim, polygon):
    # Ax + b = 0.
    A = polygon.A
    b = polygon.b
    num = len(A)
    hbar = np.zeros((num, dim))
    for i in range(num):
        hbar[i, :] = -1. * b[i] * A[i, :]              # nx - d = 0,    d = -b
    return hbar

def evaluate(data, ground_truth, ground_truth_total_cost, result):
    N_matrix = result.A
    d_matrix = -1 * result.b
    
    # compute total_cost
    n, m = len(data), len(N_matrix)
    dim = len(data[0])
    total_cost = 0.
    
    for i in range(n):
        temp = np.abs(N_matrix @ data[i, :] - d_matrix)
        total_cost += np.min(temp)
    
    # average_distance
    average_distance = total_cost / n
    ground_truth_average_distance = ground_truth_total_cost / float(n)
    
    # .....
    M_result = cal_hbar(dim, result)
    M_ground_truth = cal_hbar(dim, ground_truth)
    
    len_G = len(ground_truth.A)
    len_R = len(N_matrix)
    
    M_hbar_distance = np.zeros((len_R, len_G))
    total_hbar_distance = 0.
    
    for i in range(len_R):
        for j in range(len_G):
            M_hbar_distance[i][j] = np.linalg.norm(M_result[i, :] - M_ground_truth[j, :])
            
    total_hbar_distance = np.sum(np.min(M_hbar_distance, axis=1))
    
    return total_hbar_distance, total_cost, average_distance, ground_truth_average_distance
    
    
    
    
    
