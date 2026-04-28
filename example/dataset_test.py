import sys
sys.path.append("..")
from data.read_data import read_data_2D, read_data_3D, read_data_4D, read_data_5D, read_data_6D
from algorithm.initial_value import Hyperplane, Polyhedron
from algorithm.hyperplanes_fitting import HyperplanesFitting
import numpy as np
import time
from evaluate import evaluate


def bic_F(data, polyhedron, n, m, dim):
    """
    贝叶斯信息准则型指标 F：
    F = 2n·ln((2/n)·Σ_i r_i) + 2n + (m(dim+1) - 1)·ln(n)
    r_i 为第 i 个样本到最近超平面的距离（与 evaluate 中 total_cost 逐项一致）。
    """
    if n <= 0 or m <= 0 or polyhedron.A is None or len(polyhedron.A) == 0:
        return np.nan
    N_matrix = polyhedron.A
    d_matrix = -1.0 * polyhedron.b
    r_sum = 0.0
    for i in range(n):
        temp = np.abs(N_matrix @ data[i, :] - d_matrix)
        r_sum += float(np.min(temp))
    inner = (2.0 / n) * r_sum
    if inner <= 0.0:
        return np.nan
    return (
        2.0 * n * np.log(inner)
        + 2.0 * n
        + (m * (dim + 1) - 1.0) * np.log(float(n))
    )


DIM = 2
# method
# A: 1,  B: 2,  A+B: 3
METHOD = "3"
INITIAL = True
TRUE_NUM = 4
NUMBER_OF_DATA_FILES = 2

A = []
B = []
C = []
D = []
E = []
F_list = []
G_list = []

ALG = HyperplanesFitting(DIM, None, parallel = True, method = METHOD, whether_initial_value = INITIAL)

for data_file_index in range(NUMBER_OF_DATA_FILES):
    if DIM == 2:
        data, ground_turth_data, gt_distance = read_data_2D("../csv_dataset", "../csv_groundtruth", file_index = data_file_index)
    elif DIM == 3:
        data, ground_turth_data, gt_distance = read_data_3D("../csv_dataset", "../csv_groundtruth", file_index = data_file_index)
    elif DIM == 4:
        data, ground_turth_data, gt_distance = read_data_4D("../csv_dataset", "../csv_groundtruth", file_index = data_file_index)
    elif DIM == 5:
        data, ground_turth_data, gt_distance = read_data_5D("../csv_dataset", "../csv_groundtruth", file_index = data_file_index)
    elif DIM == 6:
        data, ground_turth_data, gt_distance = read_data_6D("../csv_dataset", "../csv_groundtruth", file_index = data_file_index)
    ground_truth_hyperplanes = []
    for i in range(len(ground_turth_data)):
        ground_truth_hyperplanes.append(Hyperplane(ground_turth_data[i, :DIM], ground_turth_data[i, DIM]))
    ground_truth_poly = Polyhedron(DIM, ground_truth_hyperplanes)

    ALG.set_data(data)
    t0 = time.perf_counter()
    if INITIAL:
        hps = ALG.solve(None)
    else:
        hps = ALG.solve(TRUE_NUM)
    G_list.append(time.perf_counter() - t0)

    polyhedron = Polyhedron(DIM, hps)

    n_samples = len(data)
    m_planes = len(hps)
    F = bic_F(data, polyhedron, n_samples, m_planes, DIM)
    if not np.isnan(F):
        F_list.append(F)

    total_hbar_distance, total_cost, average_distance, ground_truth_average_distance = evaluate(data, ground_truth_poly, gt_distance, polyhedron)
    if not np.isnan(total_hbar_distance):
        A.append(total_hbar_distance)
    if not np.isnan(total_cost):
        B.append(total_cost)
    C.append(average_distance)
    D.append(ground_truth_average_distance)
    E.append(len(ALG.hyperplanes))
print(
    "mean E (num hyperplanes):", np.mean(np.array(E)),
    "mean B (total_cost):", np.mean(np.array(B)),
    "mean A (hbar dist):", np.mean(np.array(A)),
)
if len(F_list) > 0:
    print("mean F (BIC-type):", np.mean(np.array(F_list)))
else:
    print("mean F (BIC-type): nan (no valid F)")
print("mean G (algorithm time, s):", np.mean(np.array(G_list)))