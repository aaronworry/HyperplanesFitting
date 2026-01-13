import sys
sys.path.append("..")
from data.read_data import read_data_2D, read_data_3D
from algorithm.initial_value import Hyperplane, Polyhedron
from algorithm.hyperplanes_fitting import HyperplanesFitting
import numpy as np
import time
from evaluate import evaluate

DIM = 2
# method
# A: 1,  B: 2,  A+B: 3
METHOD = "3"
INITIAL = True
TRUE_NUM = 4
NUMBER_OF_DATA_FILES = 20

A = []
B = []
C = []
D = []
E = []

ALG = HyperplanesFitting(DIM, None, parallel = False, method = METHOD, whether_initial_value = INITIAL)

for data_file_index in range(NUMBER_OF_DATA_FILES):
    if DIM == 2:
        data, ground_turth_data, gt_distance = read_data_2D("../csv_dataset", "../csv_groundtruth", file_index = data_file_index)
    elif DIM == 3:
        data, ground_turth_data, gt_distance = read_data_3D("../csv_dataset", "../csv_groundtruth", file_index = data_file_index)
    ground_truth_hyperplanes = []
    for i in range(len(ground_turth_data)):
        ground_truth_hyperplanes.append(Hyperplane(ground_turth_data[i, :DIM], ground_turth_data[i, DIM]))
    ground_truth_poly = Polyhedron(DIM, ground_truth_hyperplanes)

    ALG.set_data(data)
    if INITIAL:
        hps = ALG.solve(None)
    else:
        hps = ALG.solve(TRUE_NUM)
    
    
    polyhedron = Polyhedron(DIM, hps)

    total_hbar_distance, total_cost, average_distance, ground_truth_average_distance = evaluate(data, ground_truth_poly, gt_distance, polyhedron)
    if not np.isnan(total_hbar_distance):
        A.append(total_hbar_distance)
    if not np.isnan(total_cost):
        B.append(total_cost)
    C.append(average_distance)
    D.append(ground_truth_average_distance)
    E.append(len(ALG.hyperplanes))
print(np.mean(np.array(E)), np.mean(np.array(B)), np.mean(np.array(A)))