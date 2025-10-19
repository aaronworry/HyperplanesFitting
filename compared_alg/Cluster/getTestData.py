import numpy as np
import matplotlib.pyplot as plt


def getData(scal):
    k, b = [6, 2, -3, -0.8], [10, -9, 8, -7]
    x = np.array([-3 + 0.1*i for i in range(61)])
    result = []
    for i in range(4):
        y = k[i] * x + b[i] + scal * (np.random.rand(len(x)) - 0.5)
        result.append(np.vstack((x, y)))
    data = np.hstack((result[0], result[1], result[2], result[3]))
    return data
    # testData = data.T

def getData2(scal):
    k, b = [6, 2, -3, -1], [10, -9, 8, -7]
    x = []
    x.append(np.linspace(-2.3, -0.2, 41))
    x.append(np.linspace(0.6, 3.4, 57))
    x.append(np.linspace(-0.2, 3.4, 73))
    x.append(np.linspace(-2.3, 0.6, 59))
    result = []
    for i in range(4):
        y = k[i] * x[i] + b[i] + scal * (np.random.rand(len(x[i])) - 0.5)
        result.append(np.vstack((x[i], y)))
    data = np.hstack((result[0], result[1], result[2], result[3]))
    return data

def getData3(scal):
    k, b = [6, 2, -3, -1], [10, -9, 8, -7]
    x = []
    x.append(np.linspace(-2.2, -0.3, 39))
    x.append(np.linspace(0.7, 3.3, 53))
    x.append(np.linspace(-0.1, 3.3, 69))
    x.append(np.linspace(-2.2, 0.5, 55))
    result = []
    for i in range(4):
        y = k[i] * x[i] + b[i] + scal * (np.random.rand(len(x[i])) - 0.5)
        result.append(np.vstack((x[i], y)))
    data = np.hstack((result[0], result[1], result[2], result[3]))
    return data

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data = getData()
    x = data[0]
    y = data[1]
    ax.scatter(x, y)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.savefig('ellipsoid.png')