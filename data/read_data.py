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
    data_num = len(df)
    data = np.zeros((data_num, 2))
    for i in range(0, data_num):
        x = df["x"][i]
        y = df["y"][i]
        data[i-1, :] = np.array([x, y])

    gt_df = pd.read_csv(gt_path, encoding='utf-8')

    hyperplane_num = len(gt_df)
    hyperplane_data = np.zeros((hyperplane_num, 3))
    for i in range(0, hyperplane_num):
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
    data_num = len(df)
    data = np.zeros((data_num, 3))
    for i in range(0, data_num):
        x = df["x"][i]
        y = df["y"][i]
        z = df["z"][i]
        data[i-1, :] = np.array([x, y, z])

    gt_df = pd.read_csv(gt_path, encoding='utf-8')
    hyperplane_num = len(gt_df)
    hyperplane_data = np.zeros((hyperplane_num, 4))
    for i in range(0, hyperplane_num):
        n1 = gt_df["one"][i]
        n2 = gt_df["two"][i]
        n3 = gt_df["three"][i]
        d = gt_df["d"][i]
        hyperplane_data[i-1, :] = np.array([n1, n2, n3, d])
        total_distance = gt_df["totaldistance"][i]

    return data, hyperplane_data, total_distance


def read_data_4D(data_file, gt_file, file_index=1):
    """读取 4D 点云与超平面真值（列名 x,y,z,w 与 one..four,d,totaldistance）。"""
    data_path = data_file + "/" + str(file_index) + ".csv"
    gt_path = gt_file + "/" + str(file_index) + ".csv"
    total_distance = 0.0
    df = pd.read_csv(data_path, encoding="utf-8")
    data_num = len(df)
    data = np.zeros((data_num, 4))
    for i in range(data_num):
        data[i, :] = np.array(
            [df["x"][i], df["y"][i], df["z"][i], df["w"][i]], dtype=np.float64
        )

    gt_df = pd.read_csv(gt_path, encoding="utf-8")
    hyperplane_num = len(gt_df)
    hyperplane_data = np.zeros((hyperplane_num, 5))
    for i in range(hyperplane_num):
        hyperplane_data[i, :] = np.array(
            [
                gt_df["one"][i],
                gt_df["two"][i],
                gt_df["three"][i],
                gt_df["four"][i],
                gt_df["d"][i],
            ],
            dtype=np.float64,
        )
        total_distance = gt_df["totaldistance"][i]

    return data, hyperplane_data, total_distance


def read_data_5D(data_file, gt_file, file_index=1):
    """读取 5D 点云与超平面真值（列名 x,y,z,w,u 与 one..five,d,totaldistance）。"""
    data_path = data_file + "/" + str(file_index) + ".csv"
    gt_path = gt_file + "/" + str(file_index) + ".csv"
    total_distance = 0.0
    df = pd.read_csv(data_path, encoding="utf-8")
    data_num = len(df)
    data = np.zeros((data_num, 5))
    for i in range(data_num):
        data[i, :] = np.array(
            [df["x"][i], df["y"][i], df["z"][i], df["w"][i], df["u"][i]],
            dtype=np.float64,
        )

    gt_df = pd.read_csv(gt_path, encoding="utf-8")
    hyperplane_num = len(gt_df)
    hyperplane_data = np.zeros((hyperplane_num, 6))
    for i in range(hyperplane_num):
        hyperplane_data[i, :] = np.array(
            [
                gt_df["one"][i],
                gt_df["two"][i],
                gt_df["three"][i],
                gt_df["four"][i],
                gt_df["five"][i],
                gt_df["d"][i],
            ],
            dtype=np.float64,
        )
        total_distance = gt_df["totaldistance"][i]

    return data, hyperplane_data, total_distance


def read_data_6D(data_file, gt_file, file_index=1):
    """读取 6D 点云与超平面真值（列名 x,y,z,w,u,v 与 one..six,d,totaldistance）。"""
    data_path = data_file + "/" + str(file_index) + ".csv"
    gt_path = gt_file + "/" + str(file_index) + ".csv"
    total_distance = 0.0
    df = pd.read_csv(data_path, encoding="utf-8")
    data_num = len(df)
    data = np.zeros((data_num, 6))
    for i in range(data_num):
        data[i, :] = np.array(
            [
                df["x"][i],
                df["y"][i],
                df["z"][i],
                df["w"][i],
                df["u"][i],
                df["v"][i],
            ],
            dtype=np.float64,
        )

    gt_df = pd.read_csv(gt_path, encoding="utf-8")
    hyperplane_num = len(gt_df)
    hyperplane_data = np.zeros((hyperplane_num, 7))
    for i in range(hyperplane_num):
        hyperplane_data[i, :] = np.array(
            [
                gt_df["one"][i],
                gt_df["two"][i],
                gt_df["three"][i],
                gt_df["four"][i],
                gt_df["five"][i],
                gt_df["six"][i],
                gt_df["d"][i],
            ],
            dtype=np.float64,
        )
        total_distance = gt_df["totaldistance"][i]

    return data, hyperplane_data, total_distance
