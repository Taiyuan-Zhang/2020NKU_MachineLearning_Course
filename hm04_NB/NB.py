# -*- coding: UTF-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt

f = open('wine.data', 'r')
types = [[], [], []]                      # 按类分的所有数据
test_data = [[], [], []]
train_data = [[], [], []]
data_num = 0                            # 数据总数
test_len = []                           # 测试集里每一类的个数
means = [[], [], []]                      # 每一类的均值
std = [[], [], []]                        # 每一类的标准差
myline = '1'
while myline:
    myline = f.readline().split(',')
    if len(myline) != 14:
        break
    for t in range(len(myline)):
        if t == 0:
            myline[t] = int(myline[t])
        else:
            myline[t] = float(myline[t])
    temp = myline.pop(0)
    types[temp - 1].append(myline)
test_len = [round(len(types[i]) / 10) for i in range(3)]
data_num = sum([len(types[i]) for i in range(3)])
TP = [0, 0, 0]
FN = [0, 0, 0]
FP = [0, 0, 0]
TN = [0, 0, 0]
precision = [0, 0, 0]
recall = [0, 0, 0]
fp_rate = [0]
tp_rate = [0]
F_measure = [0, 0, 0]
accuracy = [0, 0, 0]
threshold = 1e-09


def bayes_classificate():
    for i in range(3):
        means[i] = np.mean(train_data[i], axis=0)        # 分别计算三个类别的均值
        std[i] = np.std(train_data[i], axis=0)           # 这里是标准差
    wrong_num = 0
    for i in range(3):
        for t in test_data[i]:                  # 两层循环：从每一类取每一个测试样本
            my_type = []
            for j in range(3):
                # 由于数据集中所有的属性都是连续值，连续值的似然估计可以按照高斯分布来计算：
                temp = np.log((2*math.pi) ** 0.5 * std[j])
                temp += np.power(t - means[j], 2) / (2 * np.power(std[j], 2))
                temp = np.sum(temp)
                temp = -1*temp+math.log(len(types[j])/data_num)
                my_type.append(temp)                        # 这里将所有score保存
            pre_type = my_type.index(max(my_type))          # 取分值最大的为预测类别
            FP[pre_type] += 1
            if math.exp(max(my_type)) > threshold:
                tp_rate.append(tp_rate[-1] + 0.007755)
                fp_rate.append(fp_rate[-1])
                if pre_type == i:
                    TP[i] += 1
            else:
                fp_rate.append(fp_rate[-1] + 0.0209)
                tp_rate.append(tp_rate[-1])
                if pre_type != i:
                    FN[i] += 1
            if pre_type != i:                               # 统计错误数
                wrong_num += 1
    return wrong_num


def cross_check():
    wrong_num = 0
    for i in range(10):        # 十折交叉，并且对每一类数据分层
        for j in range(3):
            if (i+1)*test_len[j] > len(types[j]):
                test_data[j] = np.mat(types[j][i*test_len[j]:])
                train_data[j] = np.mat(types[j][:i*test_len[j]])
            else:
                test_data[j] = np.mat(types[j][i*test_len[j]:(i+1)*test_len[j]])
                train_data[j] = np.mat(types[j][:i*test_len[j]]+types[j][(i+1)*test_len[j]:])
        wrong_num += bayes_classificate()
    print("分类准确率: "+str(1-wrong_num/data_num))


if __name__ == '__main__':
    cross_check()
    not1 = FP[1] + FP[2]
    not2 = FP[0] + FP[2]
    not3 = FP[0] + FP[1]
    TN[0] = not1 - FN[0]
    TN[1] = not2 - FN[1]
    TN[2] = not3 - FN[2]
    for i in range(3):
        FP[i] = FP[i] - TP[i]
        precision[i] = TP[i] / (TP[i] + FP[i])
        recall[i] = TP[i] / (TP[i] + FN[i])
        accuracy[i] = (TP[i] + TN[i]) / (TP[i] + FN[i] + FP[i] + TN[i])
        F_measure[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    print("TP: ", TP)
    print("FN: ", FN)
    print("FP: ", FP)
    print("TN: ", TN)
    for i in range(3):
        print("True class = ", i+1)
        print("fp_rate: ", fp_rate[i], "tp_rate: ", tp_rate[i])
        print("precision: ", precision[i], "recall: ", recall[i])
        print("accuracy: ", accuracy[i], "F_measure: ", F_measure[i])
    plt.plot(fp_rate, tp_rate, 'b')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.savefig('ROC.png')
    plt.show()

