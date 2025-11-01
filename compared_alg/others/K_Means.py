import numpy as np
import matplotlib.pyplot as plt



class Kmeans():
    def __init__(self, n, dim):
        self.n = n
        self.dim = dim
        self.vectors = np.zeros((n, dim))
        self.distances = np.array([0.]*self.n)
        
        self.data = None
        self.number = 0
        
    def set_data(self, data):
        self.data = data
        self.number = len(data)
        
        
    def solve(self):
        cluster = None
        beta = np.random.randn(self.n, self.dim)
        kkk = 0
        while kkk < 15:
            kkk += 1
            cluster = self.get_n_cluster(beta)
            for i in range(self.n):
                beta[i] = self.min_distance(np.array(cluster[i]))
        
        for k in range(self.n):
            beta = self.min_distance(np.array(cluster[k]))
            norm_v = np.sqrt(beta[1]**2 + 1.)
            self.distances[k] = -beta[0] / norm_v
            self.vectors[k, 0] = beta[1] / norm_v
            self.vectors[k, 1] = -1. / norm_v

        
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


    def get_n_cluster(self, beta):
        cluster = [[] for _ in range(self.n)]
        for i in range(self.number):
            index = np.argmin([np.abs(self.data[i,0] * item[1] + item[0] - self.data[i,1]) / np.sqrt(1 + item[1] ** 2) for item in beta])
            cluster[index].append(self.data[i,:])
        return cluster





