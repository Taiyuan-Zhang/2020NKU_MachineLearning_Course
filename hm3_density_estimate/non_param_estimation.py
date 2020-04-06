import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors.kde import KernelDensity
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
    # 十折交叉验证
    return X, Y


# sklearn实现的核
kernels = ('gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear')


def cross_validate_np(X, Y, kernel, bandwidth):
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # 根据分布得到模型
        pattern = []
        for i in range(1, 4):
            index = np.where(y_train == i)
            pattern.append(KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(x_train[index]))
        right_cnt = 0
        # 根据模型预测可能性
        possible = np.array([pat.score_samples(x_test) for pat in pattern]).T
        for pos, label in zip(possible, y_test):
            ret, result = pos[0], 1
            # 将概率最大的作为预测值
            for i in (1, 2):
                if ret < pos[i]:
                    ret, result = pos[i], i + 1
            if label == result:
                right_cnt += 1
        yield right_cnt / y_test.shape[0]


def fit_bandwidth(X, Y, kernel):
    ret, result = 0, []  # result保存效果最好的bandwidth
    acc_list, bands = [], [round(i, 2) for i in np.arange(0.01, 2.01, 0.01)]
    for i in bands:
        acc = np.array(list(cross_validate_np(X, Y, kernel, i))).mean()
        acc_list.append(acc)  # 保存每个bandwidth下的正确率，用于画图
        if ret < acc:  # 更新result的值
            ret, result = acc, [i]
        elif ret == acc:
            result.append(i)
    return ret, result


if __name__ == '__main__':
    # 加载数据
    (X, Y) = loadData()
    print("-----------核函数方法----------")
    # 对每个核函数，使用十折交叉验证进行验证
    for kernel in kernels:
        print('kernel = %s' % kernel)
        ret, result = fit_bandwidth(X, Y, kernel)
        print("分类性能结果：", ret)
        print("最优的平滑参数h：", result)
