from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt

from getTestData import getData, getData2, getData3

# 需要大修该算法


def get_cluster(data, label):
    cluster = [[] for _ in range(max(label) - min(label) + 1)]
    for i in range(len(data[0])):
        cluster[label[i]].append(data[:, i])
    return cluster


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
    clustering = AgglomerativeClustering(n_clusters=4, linkage='average').fit(data.T)
    labels = clustering.labels_
    cluster = get_cluster(data, labels)

    fig = plt.figure()
    bx = fig.add_subplot(121)
    x = data[0]
    y = data[1]
    bx.scatter(x, y, color='b')
    ax = fig.add_subplot(122)
    colors = ['r', 'g', 'b', 'y', 'm', 'k', 'c']
    result = len(cluster)
    print(result)
    for i in range(result):
        ax.scatter(np.array(cluster[i]).T[0], np.array(cluster[i]).T[1], color=colors[i%7])
        # beta[i] = least_squares(2, np.array(cluster[i]).T)
        beta = min_distance(2, np.array(cluster[i]).T)
        X = [-0.5, 0.5]
        y = [beta[0] - 0.5 * beta[1], beta[0] + 0.5 * beta[1]]
        ax.plot(X, y, color=colors[i%7])

    plt.show()





