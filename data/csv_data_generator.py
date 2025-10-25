# 根据generator生成一系列csv文件
import sys
sys.path.append("..")
import csv
import numpy as np
from data.random_data import Random

def genertor(dim, num, hyperplanes_num, max_distance_from_hyperplane, min_points_on_hyperplane, max_points_on_hyperplane):
    generator = Random(dim, hyperplanes_num, max_distance_from_hyperplane = max_distance_from_hyperplane, min_points_on_hyperplane = min_points_on_hyperplane, max_points_on_hyperplane = max_points_on_hyperplane, X_limit=[-5., 5.], Y_limit=[-5., 5.])
    # np.random.seed(42)
    for i in range(num):
        data, gt_distance, vectors, distances = generator.get_data()
        data_num = len(data)
        hyperplane_num = len(vectors)
        data_path = "../csv_dataset/" + str(i) + ".csv"
        gt_path = "../csv_groundtruth/" + str(i) + ".csv"
        
        with open(data_path, "w+", newline="") as file:
            csv_file = csv.writer(file)
            temp = ["x", "y"]
            csv_file.writerow(temp)
            for j in range(data_num):
                temp = [data[j][0], data[j][1]]
                csv_file.writerow(temp)
                
        with open(gt_path, "w+", newline="") as file:
            csv_file = csv.writer(file)
            temp = ["one", "two", "d", "totaldistance"]
            csv_file.writerow(temp)
            for j in range(hyperplane_num):
                temp = [vectors[j][0], vectors[j][1], distances[j], gt_distance[0]]
                csv_file.writerow(temp)
            
        
        
        
        
        
if __name__ == "__main__":
    genertor(2, 20, 4, 0.1, 30, 30)
    # genertor(2, 20, 2, 0.1, 60, 60)
    # genertor(2, 20, 3, 0.1, 40, 40)
    # genertor(2, 20, 5, 0.1, 24, 24)
    # genertor(2, 20, 6, 0.1, 20, 20)