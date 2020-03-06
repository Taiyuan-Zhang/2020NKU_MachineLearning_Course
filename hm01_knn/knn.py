import os
import numpy as np
from sklearn import neighbors
import sklearn.model_selection
import matplotlib.pyplot as plt


# 获取类标签
def getLabel(hot, size):
	for i in range(size):
		if hot[i] == '1':
			return i
	return -1


# 读训练集和测试集数据，并进行归一化处理
def loadData():
	# 训练集
	train = []
	with open(os.path.dirname(__file__) + '/semeion_train.csv', 'r') as train_file:
		for row in train_file:
			line = row.strip().split(' ')
			train.append(line[:-10] + [getLabel(line[-10:], 10)])
	# 归一化处理
	train = np.array(train, dtype=float)
	m, n = np.shape(train)
	data = train[:, 0:n - 1]
	min_data = data.min(0)
	max_data = data.max(0)
	data = (data - min_data) / (max_data - min_data)
	train[:, 0:n - 1] = data
	# 测试集
	test = []
	with open(os.path.dirname(__file__) + '/semeion_test.csv', 'r') as test_file:
		for row in test_file:
			line = row.strip().split(' ')
			test.append(line[:-10] + [getLabel(line[-10:], 10)])
	# 归一化处理
	test = np.array(test, dtype=float)
	m, n = np.shape(test)
	data = test[:, 0:n - 1]
	min_data = data.min(0)
	max_data = data.max(0)
	data = (data - min_data) / (max_data - min_data)
	test[:, 0:n - 1] = data
	return train, test


# 计算欧式距离
def euclidean_Dist(data1, data2):
	det = (data1 - data2)[:, 0:n - 1]
	return np.linalg.norm(det, axis=1)


# 计算测试集样本的k近邻，标签
def getKNNPredictedLabel(train_data, test_data, train_label, test_label, k):
	# 通过K近邻分类，获取测试集的预测标签
	correct_cnt = 0
	pre_label = []
	i = 0
	for test in test_data:
		dist = euclidean_Dist(train_data, test)
		distSorted = np.argsort(dist)
		classCount = {}
		for num in range(k):
			voteLabel = train_label[distSorted[num]]
			classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
		sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
		predictedLabel = sortedClassCount[0][0]
		pre_label.append(predictedLabel)
		if predictedLabel == test_label[i]:
			correct_cnt += 1
		i = i + 1
	return correct_cnt, pre_label


if __name__ == '__main__':
	(train_data, test_data) = loadData()
	m, n = np.shape(train_data)
	train_label = train_data[:, -1]
	test_label = test_data[:, -1]
	print("---------------原始KNN分类结果---------------")
	for k in (1, 3, 5):
		correct_cnt = getKNNPredictedLabel(train_data, test_data, train_label, test_label, k)[0]
		acc = correct_cnt / np.shape(test_data)[0]
		print('k为', k, '时，分类正确的个数：', correct_cnt, ',分类精度：%.2f' % (acc * 100), '%')

	print("--------------sklearnKNN分类结果-------------")
	for k in (1, 3, 5):
		knn = neighbors.KNeighborsClassifier(n_neighbors=k)
		knn.fit(train_data, train_label)
		acc = knn.score(test_data, test_label)
		correct_num = acc * np.shape(test_data)[0]
		print('k为', k, '时，分类正确的个数：%0.f' % correct_num, ',分类精度：%.2f' % (acc * 100), '%')

	print("-----------原始KNN十折交叉验证选择K值----------")
	original_accs = {}
	# k取值范围
	for k in range(1, 20):
		acc = 0
		# 将训练集分十折
		for i in range(10):
			# 选一折作为测试集
			test_Data = train_data[i * int(m * 0.1):(i + 1) * int(m * 0.1)]
			test_Label = test_Data[:, -1]
			# 合并剩下的九折
			a = train_data[0:i * int(m * 0.1)]
			b = train_data[(i + 1) * int(m * 0.1):n]
			train_Data = np.vstack((a, b))
			train_Label = train_Data[:, -1]
			correct_cnt = getKNNPredictedLabel(train_Data, test_Data, train_Label, test_Label, k)[0]
			acc += correct_cnt / np.shape(test_Data)[0]
		# 精度求平均值
		original_accs[k] = acc / 10
	sortedOaccs = sorted(original_accs.items(), key=lambda x: x[1], reverse=True)
	print("最终最佳精度：%.2f" % (sortedOaccs[0][1] * 100), "%", ",最终的最佳K值：", sortedOaccs[0][0])

	print("-----------sklearn十折交叉验证选择K值----------")
	parameters = {'n_neighbors': range(1, 20)}
	knn = neighbors.KNeighborsClassifier()
	clf = sklearn.model_selection.GridSearchCV(knn, parameters, cv=10)
	clf.fit(train_data, train_label)
	print("-GridSearchCV网格搜索：最终最佳精度：%.2f" % (clf.best_score_ * 100), "%", ",最终的最佳K值:",
	      clf.best_params_.get('n_neighbors'))
	sklearn_accs = {}
	for k in range(1, 20):
		knn = neighbors.KNeighborsClassifier(n_neighbors=k)
		sklearn_accs[k] = sklearn.model_selection.cross_val_score(knn, train_data, train_label, cv=10).sum() / 10
	sortedaccs = sorted(sklearn_accs.items(), key=lambda x: x[1], reverse=True)
	print("cross_val_score方法:最终最佳精度：%.2f" % (sortedaccs[0][1] * 100), "%", ",最终的最佳K值:", sortedaccs[0][0])

	# 图表显示结果
	x1 = []
	y1 = []
	x1 += sklearn_accs.keys()
	y1 += sklearn_accs.values()
	plt.title("sklearn-10-fold cross-validation")
	plt.xticks(np.arange(0, 20, 1))
	plt.plot(x1, y1)
	plt.xlabel('K')
	plt.ylabel('Accuracy')
	plt.savefig('sklearn-10-fold cross-validation.png', dpi=600)
	plt.figure()
	plt.title("original-10-fold cross-validation")
	x2 = []
	y2 = []
	x2 += original_accs.keys()
	y2 += original_accs.values()
	plt.xticks(np.arange(0, 20, 1))
	plt.plot(x2, y2)
	plt.xlabel('K')
	plt.ylabel('Accuracy')
	plt.savefig('original-10-fold cross-validation.png', dpi=600)
	plt.figure()
	plt.title("comparison-10-fold cross-validation")
	plt.xticks(np.arange(0, 20, 1))
	plt.plot(x1, y1, label="sklearn")
	plt.plot(x2, y2, label="original")
	plt.xlabel('K')
	plt.ylabel('Accuracy')
	plt.savefig('comparison-10-fold cross-validation.png', dpi=600)