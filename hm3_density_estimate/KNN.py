import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection
from sklearn.model_selection import KFold


def loadData():
    # 读数据
    data = pd.read_csv('HWData3.csv')
    data = np.array(data)
    m, n = data.shape
    X = data[:, 0:n - 1]
    Y = data[:, -1]
    # 数据标准化
    std = StandardScaler()
    X = std.fit_transform(X)
    return X, Y


def cross_validate_knn(X, Y):
    parameters = {'n_neighbors': range(1, 20)}
    knn = neighbors.KNeighborsClassifier()
    clf = sklearn.model_selection.GridSearchCV(knn, parameters, cv=10)
    clf.fit(X, Y)
    return clf.best_score_, clf.best_params_.get('n_neighbors')


def knn_density_estimate(X, Y, k, xleftlimit, xrightlimit, xstep):
    index = np.where(Y == 1)
    X_train = X[index][:, 0]
    m = X_train.shape[0]
    x = xleftlimit
    sample = np.arange(xleftlimit, xrightlimit+xstep, xstep)
    p = np.zeros((int((xrightlimit+xstep-xleftlimit)/xstep), 1))
    while x < xrightlimit+xstep/2:
        dist = np.zeros((1, m))
        for i in range(0, m):
            dist[0][i] = abs(x - X_train[i])
        dist.sort(axis=1)
        V = dist[0][k]*2
        p[int(x/xstep)+1][0] = k/m/V
        x += xstep
    plt.plot(sample, p, "-", label="k=" + str(k))
    plt.xlabel("x")
    plt.ylabel("p")
    plt.title("knn_density_estimate")
    plt.legend(loc="best")
    plt.savefig("fig(k=%d).png" % k)
    plt.show()
    print("k=%d时的概率密度曲线图存储至fig(k=" % k+str(k)+").png")


if __name__ == '__main__':
    (X, Y) = loadData()

    print("----------最近邻决策分类----------")
    result, k = cross_validate_knn(X, Y)
    print("-GridSearchCV网格搜索(十折交叉验证):")
    print("最佳K值：", k)
    print("分类性能结果：", result)
    print("---------K-近邻概率密度估计-------")
    index = np.where(Y == 1)
    X_train = X[index][:, 0]
    print("第一类特征x1")
    print(X_train)
    for i in (1, 3, 5):
        knn_density_estimate(X, Y, i, -3, -0.01, 0.01)
