# Gaussian-MM

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal


class GMM():
    def __init__(self, n, dim):
        self.n = n
        self.dim = dim
        
        self.mu = None
        self.weight = None
        self.pi = None
        self.sigma = None
        self.cluster = None
        self.data = None
        self.number = 0
        
        self.vectors = np.zeros((n, dim))
        self.distances = np.array([0.]*self.n)
        
    def set_data(self, data):
        self.data = data
        self.number = len(data)
        
        self.mu = np.random.rand(self.n, self.dim)
        self.weight = np.ones((self.number, self.n)) / self.n
        self.pi = self.weight.sum(axis = 0) / self.weight.sum()
        self.sigma = np.zeros((self.n, self.dim, self.dim)) + 2. * np.eye(self.dim)
        
    def solve(self):
        iter_num = 0
        last_w = np.zeros((self.number, self.n))
        while iter_num <= 20 or np.max(self.weight - last_w) > 0.01:
            self.E_step()
            self.M_step()
            last_w = self.weight
            iter_num += 1
        self.cluster = self.get_n_cluster(self.n, self.weight, self.data)
        
        for k in range(self.n):
            beta = self.min_distance(np.array(self.cluster[k]))
            norm_v = np.sqrt(beta[1]**2 + 1.)
            self.distances[k] = -beta[0] / norm_v
            self.vectors[k, 0] = beta[1] / norm_v
            self.vectors[k, 1] = -1. / norm_v
        
        
        
        
    def get_n_cluster(self, n, Weight, data):
        cluster = [[] for _ in range(n)]
        for i in range(len(data)):
            index = np.argmax(Weight[i])
            cluster[index].append(data[i, :])
        return cluster
        
    def E_step(self):
        pdfs = np.zeros((self.number, self.n))
        
        for i in range(self.n):
            for j in range(self.number):
                pdfs[j, i] = self.pi[i] * multivariate_normal.pdf(self.data[j,:], self.mu[i,:], self.sigma[i,:,:])
        self.weight = pdfs / pdfs.sum(axis = 1).reshape(-1, 1)


    def M_step(self):
        for i in range(self.n):
            self.mu[i] = np.average(self.data, axis=0, weights=self.weight[:, i])
            for j in range(self.dim):
                self.sigma[i, j, j] = np.average((self.data[:, j] - self.mu[i, j]) ** 2, axis = 0, weights = self.weight[:, i])

        
    def min_distance(self, data):
        x_avr, y_avr = np.mean(data[:, 0]), np.mean(data[:, 1])
        A = 0
        B = 0
        C = 0
        for i in range(len(data)):
            x = data[i, 0] - x_avr
            y = data[i, 1] - y_avr
            A += x * y
            B += x * x - y * y
            C += -1 * x * y
        delta = np.sqrt(B * B - 4 * A * C)
        k1, k2 = (delta - B) / (2 * A), (-1 * delta - B) / (2 * A)
        beta = np.array([y_avr - k1 * x_avr, k1])        # (b, k)    y = kx + b
        return beta








