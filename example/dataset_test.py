import sys
sys.path.append("..")
from data.data_from_csv import read_data_2D, read_data_3D
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

ALG = HyperplanesFitting(DIM, None, parallel = False, method = METHOD, whether_initial_value = INITIAL)

for data_file_index in range(100):
    data, ground_turth_data, gt_distance = read_data(data_file_index)

    ground_truth_hyperplanes = []
    for i in range(len(ground_turth_data)):
        ground_truth_hyperplanes.append(Hyperplane(ground_turth_data[i, :DIM], -ground_turth_data[i, DIM]))
    ground_truth_poly = Polyhedron(DIM, ground_truth_hyperplanes)

    ALG.set_data(data)
    ALG.solve()
    polyhedron = Polyhedron(DIM, ALG.hyperplanes)

    total_hbar_distance, total_cost, average_distance, ground_truth_average_distance = evaluate(data, ground_truth_poly, gt_distance, polyhedron)
    print(total_hbar_distance, total_cost, average_distance, ground_truth_average_distance)