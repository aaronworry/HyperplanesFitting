# 直线可视为非常扁的椭圆 不能收敛
# Gaussian-MM

import numpy as np
import math
from getTestData import getData, getData2, getData3
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal



def GMM(n, dim, data):
    # init
    # Mu = np.array([[2, 0], [0, 2], [-2, 0], [0, -2]])
    Mu = np.random.rand(n, dim)
    Sigma = np.zeros((n, dim, dim)) + 2 * np.eye(dim)
    Weight = np.ones((len(data[0]), n)) / n    # [num, n]
    Pi = Weight.sum(axis=0) / Weight.sum()     # [1, n]
    iter = 0
    while iter <= 100:
        Weight= E_step(data, Mu, Sigma, Pi)
        Pi, Mu, Sigma = M_step(data, Weight)
        iter += 1
    cluster = get_n_cluster(n, Weight, data)
    return Mu, Sigma, cluster

def get_n_cluster(n, Weight, data):
    cluster = [[] for _ in range(n)]
    for i in range(len(data[0])):
        index = np.argmax(Weight[i])
        cluster[index].append(data[:, i])
    return cluster

def E_step(X, Mu, Sigma, Pi):
    """
    :param X:  data   [dim, num]
    :param Mu:        [n, dim]
    :param Sigma:       [n, dim, dim]
    :param Pi:        [1, n]
    :return:
    """
    number, n = len(X[0]), len(Pi)
    pdfs = np.zeros((number, n))
    for i in range(n):
        pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X.T, Mu[i], Sigma[i])
    Weight = pdfs / pdfs.sum(axis = 1).reshape(-1, 1)

    return Weight

def M_step(X, Weight):
    n, dim, num = Weight.shape[1], len(X), len(X[0])
    Sigma = np.zeros((n, dim, dim))
    Mu = np.zeros((n, dim))
    Pi = Weight.sum(axis=0) / Weight.sum()
    for i in range(n):
        Mu[i] = np.average(X.T, axis=0, weights=Weight[:, i])
        Sigma[i] = np.average((X.T - Mu[i]) ** 2, axis = 0, weights = Weight[:, i])

        """
        add = 0
        add_sigma = np.zeros((2, 2))
        for j in range(num):
            add += Weight[j, i]
            add_sigma += ((X.T[j] - Mu[i]) ** 2) * Weight[j, i]
        Sigma[i] = add_sigma / add
        """
    return Pi, Mu, Sigma


if __name__ == "__main__":
    def least_squares(dim, data):
        """
        最小二乘拟合直线
        :param dim:
        :param data: [dim, m]
        :return:   beta_0 + beta_1 * x = y
        """
        X, Y = data[0], data[1]
        temp = np.array([1] * len(X))
        X = np.vstack((temp, X))
        beta = np.linalg.inv(X.dot(X.T)).dot(X).dot(Y)
        return beta


    def min_distance(dim, data):
        """
        拟合直线，与直线距离最小
        """
        x_avr, y_avr = np.mean(data[0]), np.mean(data[1])
        A = 0
        B = 0
        C = 0
        for i in range(len(data[0])):
            x = data[0][i] - x_avr
            y = data[1][i] - y_avr
            A += x * y
            B += x * x - y * y
            C += -1 * x * y
        delta = np.sqrt(B * B - 4 * A * C)
        k1, k2 = (delta - B) / (2 * A), (-1 * delta - B) / (2 * A)
        beta = np.array([y_avr - k1 * x_avr, k1])
        return beta


    data = getData3(0.1)
    Mu, Sigma, cluster = GMM(4, 2, data)
    fig = plt.figure()
    bx = fig.add_subplot(121)
    x = data[0]
    y = data[1]
    bx.scatter(x, y, color='b')
    ax = fig.add_subplot(122)
    ax.scatter(np.array(cluster[0]).T[0], np.array(cluster[0]).T[1], color='r')
    ax.scatter(np.array(cluster[1]).T[0], np.array(cluster[1]).T[1], color='g')
    ax.scatter(np.array(cluster[2]).T[0], np.array(cluster[2]).T[1], color='b')
    ax.scatter(np.array(cluster[3]).T[0], np.array(cluster[3]).T[1], color='y')
    colors = ['r', 'g', 'b', 'y']
    beta = np.random.randn(4, 2)
    for i in range(len(Mu)):
        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'ls': ':'}
        vals, vecs = np.linalg.eigh(Sigma[i])
        a, b = vecs[:, 0]
        theta = np.degrees(np.arctan2(b, a))
        w, h = 2 * np.sqrt(vals)
        ellipse = Ellipse(Mu[i], w, h, angle=float(theta), **plot_args)
        ax.add_patch(ellipse)
        """
        k = math.atan2(b, a)
        t = Mu[i][1] - k * Mu[i][0]
        x = np.linspace(-2, 2, 20)
        y = k * x + t
        ax.plot(x, y, color=colors[i])
        """
        # beta[i] = least_squares(2, np.array(cluster[i]).T)
        beta[i] = min_distance(2, np.array(cluster[i]).T)
        X = [-2, 2]
        y = [beta[i][0] - 2 * beta[i][1], beta[i][0] + 2 * beta[i][1]]
        ax.plot(X, y, color=colors[i])



    plt.show()
