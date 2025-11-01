import sys
sys.path.append("..")
from data.read_data import read_data_2D, read_data_3D
from algorithm.initial_value import Hyperplane, Polyhedron
from others.GMM import GMM
from others.agglomerativeClustering import AggCluster
from others.optics import opticsC
from others.DBSCAN import dbscan
from others.K_Means import Kmeans
import numpy as np
import time
from evaluate import evaluate

DIM = 2
TRUE_NUM = 4

A = []
B = []
C = []
D = []
E = []

# ALG = GMM(TRUE_NUM, DIM)
# ALG = AggCluster(TRUE_NUM, DIM)
# ALG = opticsC(TRUE_NUM, DIM)
# ALG = dbscan(TRUE_NUM, DIM)
ALG = Kmeans(TRUE_NUM, DIM)

for data_file_index in range(20):
    data, ground_turth_data, gt_distance = read_data_2D("../csv_dataset", "../csv_groundtruth", file_index = data_file_index)

    ground_truth_hyperplanes = []
    for i in range(len(ground_turth_data)):
        ground_truth_hyperplanes.append(Hyperplane(ground_turth_data[i, :DIM], ground_turth_data[i, DIM]))
    ground_truth_poly = Polyhedron(DIM, ground_truth_hyperplanes)

    ALG.set_data(data)
    ALG.solve()
    hps = []
    for i in range(ALG.n):
        hps.append(Hyperplane(ALG.vectors[i, :], ALG.distances[i]))
    polyhedron = Polyhedron(DIM, hps)
    
    print(ALG.vectors)

    total_hbar_distance, total_cost, average_distance, ground_truth_average_distance = evaluate(data, ground_truth_poly, gt_distance, polyhedron)
    if not np.isnan(total_hbar_distance):
        A.append(total_hbar_distance)
    if not np.isnan(total_cost):
        B.append(total_cost)
    C.append(average_distance)
    D.append(ground_truth_average_distance)
    E.append(ALG.n)
print(np.mean(np.array(E)), np.mean(np.array(B)), np.mean(np.array(A)))