# 根据generator生成一系列csv文件
import sys
import os
sys.path.append("..")
import csv
import numpy as np
from data.random_data import Random

def genertor(dim, num, hyperplanes_num, max_distance_from_hyperplane, min_points_on_hyperplane, max_points_on_hyperplane, 
             data_dir=None, gt_dir=None):
    """
    生成超平面拟合数据集
    
    Args:
        dim: 数据维度 (2, 3, 4, 5 或 6)
        num: 生成样本数量
        hyperplanes_num: 每个样本的超平面数量
        max_distance_from_hyperplane: 点到超平面的最大噪声距离
        min_points_on_hyperplane: 每个超平面上的最少点数
        max_points_on_hyperplane: 每个超平面上的最多点数
        data_dir: 数据输出目录 (默认根据维度选择 csv_dataset 或 csv_dataset_3d)
        gt_dir: 真值输出目录 (默认根据维度选择 csv_groundtruth 或 csv_groundtruth_3d)
    """
    # 设置默认输出目录
    if data_dir is None:
        if dim == 2:
            data_dir = "../csv_dataset"
        elif dim == 3:
            data_dir = "../csv_dataset_3d"
        else:
            data_dir = f"../csv_dataset"
    if gt_dir is None:
        if dim == 2:
            gt_dir = "../csv_groundtruth"
        elif dim == 3:
            gt_dir = "../csv_groundtruth_3d"
        else:
            gt_dir = f"../csv_groundtruth"
    
    # 确保目录存在
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    
    generator = Random(dim, hyperplanes_num, max_distance_from_hyperplane = max_distance_from_hyperplane, min_points_on_hyperplane = min_points_on_hyperplane, max_points_on_hyperplane = max_points_on_hyperplane, X_limit=[-5., 5.], Y_limit=[-5., 5.])
    # np.random.seed(42)
    for i in range(num):
        data, gt_distance, vectors, distances = generator.get_data()
        gt_total = float(np.asarray(gt_distance).ravel()[0])
        data_num = len(data)
        hyperplane_num = len(vectors)
        data_path = os.path.join(data_dir, str(i) + ".csv")
        gt_path = os.path.join(gt_dir, str(i) + ".csv")
        if dim == 2:
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
                    temp = [vectors[j][0], vectors[j][1], distances[j], gt_total]
                    csv_file.writerow(temp)
        
        elif dim == 3:
            with open(data_path, "w+", newline="") as file:
                csv_file = csv.writer(file)
                temp = ["x", "y", "z"]
                csv_file.writerow(temp)
                for j in range(data_num):
                    temp = [data[j][0], data[j][1], data[j][2]]
                    csv_file.writerow(temp)
                    
            with open(gt_path, "w+", newline="") as file:
                csv_file = csv.writer(file)
                temp = ["one", "two", "three", "d", "totaldistance"]
                csv_file.writerow(temp)
                for j in range(hyperplane_num):
                    temp = [vectors[j][0], vectors[j][1], vectors[j][2], distances[j], gt_total]
                    csv_file.writerow(temp)

        elif dim in (4, 5, 6):
            coord_names = ["x", "y", "z", "w", "u", "v"][:dim]
            normal_names = ["one", "two", "three", "four", "five", "six"][:dim]
            with open(data_path, "w+", newline="") as file:
                csv_file = csv.writer(file)
                csv_file.writerow(coord_names)
                for j in range(data_num):
                    csv_file.writerow([data[j][k] for k in range(dim)])

            with open(gt_path, "w+", newline="") as file:
                csv_file = csv.writer(file)
                csv_file.writerow(normal_names + ["d", "totaldistance"])
                for j in range(hyperplane_num):
                    row = [vectors[j][k] for k in range(dim)]
                    row.append(distances[j])
                    row.append(gt_total)
                    csv_file.writerow(row)

        
        
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='生成超平面拟合数据集')
    parser.add_argument('--dim', type=int, default=2, choices=[2, 3, 4, 5, 6], help='数据维度')
    parser.add_argument('--num', type=int, default=10, help='生成样本数量')
    parser.add_argument('--hyperplanes', type=int, default=5, help='超平面数量')
    parser.add_argument('--noise', type=float, default=0.3, help='噪声水平')
    parser.add_argument('--min_points', type=int, default=24, help='每个超平面最少点数')
    parser.add_argument('--max_points', type=int, default=24, help='每个超平面最多点数')
    args = parser.parse_args()
    
    print(f"生成 {args.dim}D 数据集: {args.num} 个样本, 每个 {args.hyperplanes} 个超平面")
    genertor(args.dim, args.num, args.hyperplanes, args.noise, args.min_points, args.max_points)
    print("数据生成完成!")
    
    # 旧的手动配置方式 (已注释)
    # genertor(2, 20, 4, 0.1, 30, 30)
    # genertor(3, 20, 4, 0.1, 30, 30)  # 3D 数据
