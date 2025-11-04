import sys
sys.path.append("..")
from data.read_data import read_data_2D, read_data_3D
from algorithm.initial_value import Hyperplane, Polyhedron
from algorithm.hyperplanes_fitting import HyperplanesFitting
from view.plot import ResultViewer
import numpy as np
import time
from evaluate import evaluate

DIM = 2

# method
# A: 1,  B: 2,  A+B: 3
METHOD = "3"
INITIAL = True
TRUE_NUM = 4

data, hyperplane_data, gt_distance = read_data_2D("../csv_dataset", "../csv_groundtruth", file_index = 3)
ground_truth_hyperplanes = []
for i in range(len(hyperplane_data)):
        ground_truth_hyperplanes.append(Hyperplane(hyperplane_data[i, :DIM], hyperplane_data[i, DIM]))
ground_truth_poly = Polyhedron(DIM, ground_truth_hyperplanes)


ALG = HyperplanesFitting(DIM, data, parallel = False, method = METHOD, whether_initial_value = INITIAL)
if INITIAL:
    hps = ALG.solve(None)
else:
    hps = ALG.solve(TRUE_NUM)

polyhedron = Polyhedron(DIM, hps)

viewer = ResultViewer(dim = DIM, data = data, ground_truth=ground_truth_poly, initial_hyperplanes = polyhedron, convex_region_hyperplanes = None, X_limit = [-5., 5.], Y_limit = [-5., 5.])
viewer.draw_result()
viewer.show(False, True)

total_hbar_distance, total_cost, average_distance, ground_truth_average_distance = evaluate(data, ground_truth_poly, gt_distance, polyhedron)
print(total_hbar_distance, total_cost, average_distance, ground_truth_average_distance)

