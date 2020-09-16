import numpy as np
from random import randrange
from scipy.io import loadmat
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def distanceNorm(Norm, D_value):
	if Norm == '1':
		counter = np.absolute(D_value)
		counter = np.sum(counter)
	elif Norm == '2':
		counter = np.power(D_value, 2)
		counter = np.sum(counter)
		counter = np.sqrt(counter)
	elif Norm == 'Infinity':
		counter = np.absolute(D_value)
		counter = np.max(counter)
	else:
		raise Exception('......')
	return counter


def fit(features, labels, iter_ratio):
	(n_samples, n_features) = np.shape(features)
	distance = np.zeros((n_samples, n_samples))
	weight = np.zeros(n_features)

	if iter_ratio >= 0.5:
		for index_i in range(n_samples):
			for index_j in range(index_i + 1, n_samples):
				D_value = features[index_i] - features[index_j]
				distance[index_i, index_j] = distanceNorm('2', D_value)
		distance += distance.T
	else:
		pass

	for iter_num in range(int(iter_ratio * n_samples)):
		nearHit = list()
		nearMiss = list()
		distance_sort = list()

		index_i = randrange(0, n_samples, 1)
		self_features = features[index_i]

		if iter_ratio >= 0.5:
			distance[index_i, index_i] = np.max(distance[index_i])
			for index in range(n_samples):
				distance_sort.append([distance[index_i, index], index, labels[index]])
		else:
			distance = np.zeros(n_samples)
			for index_j in range(n_samples):
				D_value = features[index_i] - features[index_j]
				distance[index_j] = distanceNorm('2', D_value)
			distance[index_i] = np.max(distance)
			for index in range(n_samples):
				distance_sort.append([distance[index], index, labels[index]])
		distance_sort.sort(key=lambda x: x[0])
		for index in range(n_samples):
			if nearHit == [] and distance_sort[index][2] == labels[index_i]:
				nearHit = features[distance_sort[index][1]]
			elif nearMiss == [] and distance_sort[index][2] != labels[index_i]:
				nearMiss = features[distance_sort[index][1]]
			elif nearHit != [] and nearMiss != []:
				break
			else:
				continue

		weight = weight - np.power(self_features - nearHit, 2) + np.power(self_features - nearMiss, 2)
	return weight / (iter_ratio * n_samples)


if __name__ == '__main__':
	data = loadmat("urban.mat")
	features = data["X"]
	labels = data["Y"]
	features = normalize(X=features, norm='l2', axis=0)
	weight = fit(features, labels, 1)
	n_weight = len(weight)
	x = ["1/6", "2/6", "3/6", "4/6", "5/6"]
	accuracy = [[], [], [], []]
	AUC = [[], [], [], []]
	for i in range(1, 6):
		print("-----选择排序前%d/6的特征-----" % i)
		index = np.argsort(weight)[int(1 - (n_weight*i / 6)): -1]
		selected = features[:, index]
		x_train, x_test, y_train, y_test = train_test_split(selected, labels, random_state=0)
		y_train = y_train.ravel()
		y_test_hot = label_binarize(y_test, classes=range(1, 10))
		knn = KNeighborsClassifier(n_neighbors=3)
		knn.fit(x_train, y_train)
		knn_score = knn.predict_proba(x_test)
		knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test_hot.ravel(), knn_score.ravel())
		knn_auc = auc(knn_fpr, knn_tpr)
		knn_accuracy = knn.score(x_test, y_test)
		accuracy[0].append(knn_accuracy)
		AUC[0].append(knn_auc)
		print("KNN分类精度：", knn_accuracy)
		print("AUC值：", knn_auc)
		NB = GaussianNB()
		NB.fit(x_train, y_train)
		NB_score = NB.predict_proba(x_test)
		NB_fpr, NB_tpr, NB_thresholds = roc_curve(y_test_hot.ravel(), NB_score.ravel())
		NB_auc = auc(NB_fpr, NB_tpr)
		NB_accuracy = NB.score(x_test, y_test)
		accuracy[1].append(NB_accuracy)
		AUC[1].append(NB_auc)
		print("NB分类精度：", NB_accuracy)
		print("AUC值：", NB_auc)
		clf = svm.SVC(probability=True)
		clf.fit(x_train, y_train)
		svm_score = clf.predict_proba(x_test)
		svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test_hot.ravel(), svm_score.ravel())
		svm_auc = auc(svm_fpr, svm_tpr)
		svm_accuracy = clf.score(x_test, y_test)
		accuracy[2].append(svm_accuracy)
		AUC[2].append(svm_auc)
		print("NB分类精度：", svm_accuracy)
		print("AUC值：", svm_auc)
		rf = RandomForestClassifier()
		rf.fit(x_train, y_train)
		rf_score = rf.predict_proba(x_test)
		rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test_hot.ravel(), rf_score.ravel())
		rf_auc = auc(rf_fpr, rf_tpr)
		rf_accuracy = rf.score(x_test, y_test)
		accuracy[3].append(rf_accuracy)
		AUC[3].append(rf_auc)
		print("随机森林分类精度：", rf_accuracy)
		print("AUC值：", svm_auc)
	plt.figure()
	plt.title('relief-F_Accuracy')
	plt.plot(x, accuracy[0], color="r", linestyle=':', label='KNN', marker='o')
	plt.plot(x, accuracy[1], color="y", linestyle='--', label='NB', marker='+')
	plt.plot(x, accuracy[2], color="b", linestyle='-', label='SVM', marker='*')
	plt.plot(x, accuracy[3], color="g", linestyle='-.', label='RandomForest', marker='s')
	plt.legend(loc=4)
	plt.savefig('relief-F_Accuracy.png')
	plt.figure()
	plt.title('relief-F_AUC')
	plt.plot(x, AUC[0], color="r", linestyle=':', label='KNN', marker='o')
	plt.plot(x, AUC[1], color="y", linestyle='--', label='NB', marker='+')
	plt.plot(x, AUC[2], color="b", linestyle='-', label='SVM', marker='*')
	plt.plot(x, AUC[3], color="g", linestyle='-.', label='RandomForest', marker='s')
	plt.legend(loc=4)
	plt.savefig('relief-F_AUC.png')