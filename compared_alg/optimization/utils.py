import numpy as np
import matplotlib.pyplot as plt
import math

def generate_hyperplanes_feature(dim, number):
    # 法向量 + 点 描述超平面
    if dim == 2:
        points = np.zeros((number, dim))
        vectors = np.zeros((number, dim))
        for i in range(number):
            point = [0., 0.]
            vector = [0., 0.]
            for k in range(dim):
                point[k] = 100. * np.random.rand(1) - 50.
            norm = math.sqrt(point[0] ** 2 + point[1] ** 2)
            if norm <= 1e-5:
                theta = 2 * math.pi * np.random.rand(1) - math.pi
                vector = [math.cos(theta), math.sin(theta)]
            else:
                vector = [point[0] / norm, point[1] / norm]
            points[i, :] = np.reshape(np.array(point), (1, dim))
            vectors[i, :] = np.reshape(np.array(vector), (1, dim))
        return points, vectors
    elif dim == 3:
        pass

def general_feature_from_point_and_normalvector(points, vectors):
    dim = len(points[0])
    num = len(points)
    A = np.zeros((num, dim))
    b = np.zeros((num, 1))
    for i in range(num):
        for j in range(dim):
            A[i, j] = vectors[i, j]
            b[i, 0] -= vectors[i, j] * points[i, j]
    return A, b


def get_scatter_hyperplane(dim, point, vector, num_range = [4, 5]):
    result = np.zeros((1, dim))
    if dim == 2:
        start_x = 90. * np.random.rand(1) - 50.
        end_x = start_x + 10. + (40. - start_x) * np.random.rand(1)
        num = int(np.random.rand(1)[0] * (num_range[1] - num_range[0])) + num_range[0]
        x = np.linspace(start_x, end_x, num)
        result = np.zeros((num, dim))
        for i in range(num):
            distance = 4. * np.random.rand(1) - 2.
            # (x - x0) vec0 + (y - y0) vec1 = distance
            if vector[1] <= 1e-5 and vector[1] >= -1e-5:
                if vector[1] >= 0:
                    vector[1] = 1e-5
                else:
                    vector[1] = -1e-5
            result[i, 0] = x[i]
            result[i, 1] = point[1] + (distance - vector[0] * (x[i] - point[0])) / vector[1]
    elif dim == 3:
        pass
    
    return result
        

def get_data(number, dim, num_range=[20, 30]):
    # n x dim      n points
    points, vectors = generate_hyperplanes_feature(dim, number)
    A, b = general_feature_from_point_and_normalvector(points, vectors)
    result = np.zeros((1, dim))
    distance = np.zeros((number, 1))
    for i in range(number):
        temp = get_scatter_hyperplane(dim, points[i], vectors[i], num_range)
        result = np.vstack((result, temp))
        for j in range(dim):
            distance[i, 0] += vectors[i, j] * points[i, j]
    return A, b, vectors, points, distance, result[1:,:]
    
    
def draw(data, A, b, ground_A, ground_b):
    #所有点以及pre_hyperplances
    fig = plt.figure()
    bx = fig.add_subplot(111)
    x = data[:, 0]
    y = data[:, 1]
    bx.scatter(x, y, color='black')
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    for i in range(len(b)):
        X = [-50, 50]
        # Ax + By + b = 0
        Y = [(- b[i] - A[i, 0]*X[0]) / (A[i, 1] + 1e-5), (- b[i] - A[i, 0]*X[1]) / (A[i, 1] + 1e-5)]
        Y1 = [(- ground_b[i] - ground_A[i, 0]*X[0]) / (ground_A[i, 1] + 1e-5), (- ground_b[i] - ground_A[i, 0]*X[1]) / (ground_A[i, 1] + 1e-5)]
        bx.plot(X, Y, color=colors[i%len(colors)])
        bx.plot(X, Y1, color="black")
    # bx.scatter(0, 0, color='r')
    # for item in pre_hyperplances:
    #    x = item[0] * np.cos(item[1])
    #    y = item[0] * np.sin(item[1])
    #    bx.scatter(x, y, color='r')
    # ax = fig.add_subplot(122)
    # colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    # for i in range(len(cluster)):
        # if len(cluster[i]) > 0:
            # ax.scatter(np.array(cluster[i]).T[0], np.array(cluster[i]).T[1], color=colors[i%len(colors)])
            # beta = min_distance(np.array(cluster[i]).T)
            # X = [-2, 2]
            # y = [beta[0] - 2 * beta[1], beta[0] + 2 * beta[1]]
            # ax.plot(X, y, color=colors[i])

    plt.show()



    
    
if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data = getData(0.1)
    x = data[0]
    y = data[1]
    ax.scatter(x, y)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.savefig('ellipsoid.png')