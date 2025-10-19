import sys
sys.path.append("..")
from data.random_data import Random
from algorithm.initial_value import Hyperplane, Polyhedron
from algorithm.hyperplanes_fitting import HyperplanesFitting
from view.plot import ResultViewer
import numpy as np
import time
from evaluate import evaluate

DIM = 2
# data features
HYPERPLANES_NUM = 4
MAX_POINT_DISTANCE = 0.1
MIN_POINT_NUM = 30
MAX_POINT_NUM = 30

# method
# A: 1,  B: 2,  A+B: 3
METHOD = "3"
INITIAL = True


generator = Random(DIM, HYPERPLANES_NUM, max_distance_from_hyperplane = MAX_POINT_DISTANCE, min_points_on_hyperplane = MIN_POINT_NUM, max_points_on_hyperplane = MAX_POINT_NUM, X_limit=[-5., 5.], Y_limit=[-5., 5.])
data, gt_distance, _, _ = generator.get_data()
ground_truth_A, ground_truth_b = generator.ground_truth_A, generator.ground_truth_b
ground_truth_hyperplanes = []
for i in range(len(ground_truth_b)):
    ground_truth_hyperplanes.append(Hyperplane(ground_truth_A[i], -ground_truth_b[i]))
ground_truth_poly = Polyhedron(DIM, ground_truth_hyperplanes)


ALG = HyperplanesFitting(DIM, data, parallel = False, method = METHOD, whether_initial_value = INITIAL)
ALG.solve()
polyhedron = Polyhedron(DIM, ALG.hyperplanes)


viewer = ResultViewer(dim = DIM, data = data, ground_truth=ground_truth_poly, initial_hyperplanes = polyhedron, convex_region_hyperplanes = None, X_limit = [-5., 5.], Y_limit = [-5., 5.])
viewer.draw_result()
viewer.show(True, True)

total_hbar_distance, total_cost, average_distance, ground_truth_average_distance = evaluate(data, ground_truth_poly, gt_distance, polyhedron)
print(total_hbar_distance, total_cost, average_distance, ground_truth_average_distance)

