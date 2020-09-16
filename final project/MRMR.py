import numpy as np
from scipy.io import loadmat
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize, label_binarize
import matplotlib.pyplot as plt


def entropy(c):
	c_normalized = c / float(np.sum(c))
	c_normalized = c_normalized[np.nonzero(c_normalized)]
	H = -sum(c_normalized * np.log2(c_normalized))
	return H

def feature_label_MIs(arr, y):
	m, n = arr.shape
	MIs = []
	p_y = np.histogram(y)[0]
	h_y = entropy(p_y)

	for i in range(n):
		p_i = np.histogram(arr[:, i])[0]
		p_iy = np.histogram2d(arr[:, 0], y)[0]

		h_i = entropy(p_i)
		h_iy = entropy(p_iy)

		MI = h_i + h_y - h_iy
		MIs.append(MI)
	return MIs


def feature_feature_MIs(x, y):
	p_x = np.histogram(x)[0]
	p_y = np.histogram(y)[0]
	p_xy = np.histogram2d(x, y)[0]
	h_x = entropy(p_x)
	h_y = entropy(p_y)
	h_xy = entropy(p_xy)
	return h_x + h_y - h_xy


def fit(X, y, feature_n):
	feature_num = feature_n

	MIs = feature_label_MIs(X, y)
	max_MI_arg = np.argmax(MIs)

	selected_features = []

	MIs = list(zip(range(len(MIs)), MIs))
	selected_features.append(MIs.pop(int(max_MI_arg)))

	while True:
		max_theta = float("-inf")
		max_theta_index = None
		for mi_outset in MIs:
			ff_mis = []
			for mi_inset in selected_features:
				ff_mi = feature_feature_MIs(X[:, mi_outset[0]], X[:, mi_inset[0]])
				ff_mis.append(ff_mi)
			theta = mi_outset[1] - 1 / len(selected_features) * sum(ff_mis)
			if theta >= max_theta:
				max_theta = theta
				max_theta_index = mi_outset
		selected_features.append(max_theta_index)
		MIs.remove(max_theta_index)
		if len(selected_features) == feature_num:
			break

	selected_features = [ind for ind, mi in selected_features]
	return selected_features


if __name__ == '__main__':
	data = loadmat("urban.mat")
	features = data["X"]
	labels = data["Y"]
	features = normalize(X=features, norm='l2', axis=0)
	weight = fit(features, labels.ravel(), 147)
	n_weight = len(weight)
	x = ["1/6", "2/6", "3/6", "4/6", "5/6"]
	accuracy = [[], [], [], []]
	AUC = [[], [], [], []]
	for i in range(1, 6):
		print("-----选择排序前%d/6的特征-----" % i)
		index = weight[0: int(n_weight*i / 6)]
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
	plt.title('MRMR_Accuracy')
	plt.plot(x, accuracy[0], color="r", linestyle=':', label='KNN', marker='o')
	plt.plot(x, accuracy[1], color="y", linestyle='--', label='NB', marker='+')
	plt.plot(x, accuracy[2], color="b", linestyle='-', label='SVM', marker='*')
	plt.plot(x, accuracy[3], color="g", linestyle='-.', label='RandomForest', marker='s')
	plt.legend(loc=4)
	plt.savefig("MRMR_Accuracy.png")
	plt.figure()
	plt.title('MRMR_AUC')
	plt.plot(x, AUC[0], color="r", linestyle=':', label='KNN', marker='o')
	plt.plot(x, AUC[1], color="y", linestyle='--', label='NB', marker='+')
	plt.plot(x, AUC[2], color="b", linestyle='-', label='SVM', marker='*')
	plt.plot(x, AUC[3], color="g", linestyle='-.', label='RandomForest', marker='s')
	plt.legend(loc=4)
	plt.savefig("MRMR_AUC.png")
