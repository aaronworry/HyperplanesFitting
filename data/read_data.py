import math
import numpy as np
import re
import pandas as pd
import time

def read_data_2D(data_file, gt_file, file_index = 1):
    data_path = data_file + "/" + str(file_index) + ".csv"
    gt_path = gt_file +  "/" + str(file_index) + ".csv"
    total_distance = 0.
    
    df = pd.read_csv(data_path, encoding='utf-8')
    data_num = len(df) - 1
    data = np.zeros((data_num, 2))
    for i in range(1, data_num + 1):
        x = df["x"][i]
        y = df["y"][i]
        data[i-1, :] = np.array([x, y])
        
    gt_df = pd.read_csv(gt_path, encoding='utf-8')
    hyperplane_num = len(gt_df) - 1
    hyperplane_data = np.zeros((hyperplane_num, 3))
    for i in range(1, hyperplane_num + 1):
        n1 = gt_df["one"][i]
        n2 = gt_df["two"][i]
        d = gt_df["d"][i]
        hyperplane_data[i-1, :] = np.array([n1, n2, d])
        total_distance = gt_df["totaldistance"][i]
    
    return data, hyperplane_data, total_distance
    
    

def read_data_3D(data_file, gt_file, file_index = 1):
    data_path = data_file + "/" + str(file_index) + ".csv"
    gt_path = gt_file +  "/" + str(file_index) + ".csv"
    total_distance = 0.
    df = pd.read_csv(data_path, encoding='utf-8')
    data_num = len(df) - 1
    data = np.zeros((data_num, 2))
    for i in range(1, data_num + 1):
        x = df["x"][i]
        y = df["y"][i]
        z = df["z"][i]
        data[i-1, :] = np.array([x, y, z])
        
    gt_df = pd.read_csv(gt_path, encoding='utf-8')
    hyperplane_num = len(gt_df) - 1
    hyperplane_data = np.zeros((hyperplane_num, 3))
    for i in range(1, hyperplane_num + 1):
        n1 = gt_df["one"][i]
        n2 = gt_df["two"][i]
        n3 = gt_df["three"][i]
        d = gt_df["d"][i]
        hyperplane_data[i-1, :] = np.array([n1, n2, n3, d])
        total_distance = gt_df["totaldistance"][i]
    
    return data, hyperplane_data, total_distance
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
