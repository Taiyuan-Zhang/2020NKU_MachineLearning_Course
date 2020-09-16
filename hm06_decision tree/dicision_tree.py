import numpy as np
import codecs
from sklearn.metrics import accuracy_score

feature_dict = {"色泽": ["青绿", "乌黑", "浅白"],
                "根蒂": ["蜷缩", "稍蜷", "硬挺"],
                "敲声": ["浊响", "沉闷", "清脆"],
                "纹理": ["清晰", "稍糊", "模糊"]
                }
lable_list = ["否", "是"]
feature_list = ["色泽", "根蒂", "敲声", "纹理"]


def load_txt1(path):
	ans = []
	with codecs.open(path, "r", "GBK") as f:
		line = f.readline()
		line = f.readline()
		while line:
			d = line.rstrip("\r\n").split(',')
			# print(d)
			re = []
			# 输入编号方便追踪
			re.append(int(d[0]))
			re.append(feature_dict.get("色泽").index(d[1]))
			re.append(feature_dict.get("根蒂").index(d[2]))
			re.append(feature_dict.get("敲声").index(d[3]))
			re.append(feature_dict.get("纹理").index(d[4]))

			re.append(lable_list.index(d[-1]))
			ans.append(np.array(re))
			line = f.readline()
	return np.array(ans)


def load_txt2(path):
	ans = []
	with codecs.open(path, "r", "GBK") as f:
		line = f.readline()
		line = f.readline()
		while line:
			d = line.rstrip("\r\n").split(',')
			# print(d)
			re = []
			# 输入编号方便追踪
			re.append(int(d[0]))
			re.append(feature_dict.get("色泽").index(d[1]))
			re.append(feature_dict.get("根蒂").index(d[2]))
			re.append(feature_dict.get("敲声").index(d[3]))
			re.append(feature_dict.get("纹理").index(d[4]))
			re.append(float(d[5]))
			re.append(lable_list.index(d[-1]))
			ans.append(np.array(re))
			line = f.readline()
	return np.array(ans)


class Node:
	def __init__(self, attr, label, v):
		# label == pi是非叶节点
		# attr == pi 是叶节点
		self.attr = attr
		self.label = label
		self.attr_v = v
		self.children = []


def is_same_on_attr(X, attrs):  # 验证属性上的值是否均相同
	X_a = X[:, attrs]
	target = X_a[0]
	for r in range(X_a.shape[0]):
		row = X_a[r]
		if (row != target).any():
			return False
	return True


def ent(D):
	# D is a 1d np array which actually is Y
	s = 0
	for k in set(D):
		p_k = np.sum(np.where(D == k, 1, 0)) / np.shape(D)[0]
		if p_k == 0:
			# 此时Pklog2Pk 定义为 0
			continue
		s += p_k * np.log2(p_k)
	return -s


def gain(X, Y, attr):
	# X, Y 是numpy arrary attr是某个特征的index
	x_attr_col = X[:, attr]
	ent_Dv = []
	weight_Dv = []
	# 离散值处理
	if attr != 4:
		for x_v in set(x_attr_col):
			index_x_equal_v = np.where(x_attr_col == x_v)
			y_x_equal_v = Y[index_x_equal_v]
			ent_Dv.append(ent(y_x_equal_v))
			weight_Dv.append(np.shape(y_x_equal_v)[0] / np.shape(Y)[0])
	else:
		sort_attr = sorted(set(x_attr_col))
		T_attr = []
		for i in range(len(sort_attr) - 1):
			T_attr.append((sort_attr[i] + sort_attr[i + 1]) / 2)
		maxGain = 0.0
		bestPoint = -1
		for i in range(len(sort_attr) - 1):
			Label0 = []  # 用于存放小于划分点的值
			Label1 = []  # 用于存放大于划分点的值
			for j in range(len(x_attr_col)):
				if x_attr_col[j] < T_attr[i]:
					Label0.append(Y[j])
				else:
					Label1.append(Y[j])
			ent_Dv.append(ent(Label0))
			ent_Dv.append(ent(Label1))
			weight_Dv.append(np.shape(Label0)[0] / np.shape(Y)[0])
			weight_Dv.append(np.shape(Label1)[0] / np.shape(Y)[0])
			Gain = ent(Y) - np.sum(np.array(ent_Dv) * np.array(weight_Dv))
			if Gain > maxGain:
				maxGain = Gain
				bestPoint = T_attr[i]
		return maxGain, bestPoint

	return ent(Y) - np.sum(np.array(ent_Dv) * np.array(weight_Dv))


def IV(X, Y, attr):
	x_attr_col = X[:, attr]
	weight_Dv = []
	for x_v in set(x_attr_col):
		index_x_equal_v = np.where(x_attr_col == x_v)
		y_x_equal_v = Y[index_x_equal_v]
		weight_Dv.append(np.shape(y_x_equal_v)[0] / np.shape(Y)[0])
	return np.sum(np.array(weight_Dv) * np.log2(weight_Dv))


def gain_ratio(X, Y, attr):
	if attr == 4:
		maxgain, bestpoint = gain(X, Y, attr)
		return maxgain / (-IV(X, Y, attr)), bestpoint
	return gain(X, Y, attr) / (-IV(X, Y, attr))


def dicision_tree_init_ID3(X, Y, attrs, root, purity_cal):
	# 递归基
	if len(set(Y)) == 1:
		root.attr = np.pi
		root.label = Y[0]
		return None

	if len(attrs) == 0 or is_same_on_attr(X, attrs):
		root.attr = np.pi
		# Y 中出现次数最多的label设定为node的label
		root.label = np.argmax(np.bincount(Y))
		return None

	# 计算每个attr的划分收益
	purity_attrs = []
	for i, a in enumerate(attrs):
		p = purity_cal(X, Y, a)
		purity_attrs.append(p)
	# print(purity_attrs)
	chosen_index = purity_attrs.index(max(purity_attrs))
	chosen_attr = attrs[chosen_index]

	root.attr = chosen_attr
	root.label = np.pi

	del attrs[chosen_index]

	x_attr_col = X[:, chosen_attr]
	# 离散数据处理
	for x_v in set(X[:, chosen_attr]):
		n = Node(-1, -1, x_v)
		root.children.append(n)
		# 不可能Dv empty 要是empty压根不会在set里
		# 选出 X[attr] == x_v的行

		index_x_equal_v = np.where(x_attr_col == x_v)
		X_x_equal_v = X[index_x_equal_v]
		Y_x_equal_v = Y[index_x_equal_v]
		dicision_tree_init_ID3(X_x_equal_v, Y_x_equal_v, attrs, n, purity_cal)


def dicision_tree_init_C4_5(X, Y, attrs, root, purity_cal):
	# 递归基
	if len(set(Y)) == 1:
		root.attr = np.pi
		root.label = Y[0]
		return None

	if len(attrs) == 0 or is_same_on_attr(X, attrs):
		root.attr = np.pi
		# Y 中出现次数最多的label设定为node的label
		root.label = np.argmax(np.bincount(Y))
		return None

	# 计算每个attr的划分收益
	purity_attrs = []
	for i, a in enumerate(attrs):
		if a == 4:
			p, point = purity_cal(X, Y, a)
		else:
			p = purity_cal(X, Y, a)
		purity_attrs.append(p)
	# print(purity_attrs)
	chosen_index = purity_attrs.index(max(purity_attrs))
	chosen_attr = attrs[chosen_index]

	root.attr = chosen_attr
	root.label = np.pi

	del attrs[chosen_index]

	x_attr_col = X[:, chosen_attr]
	if chosen_index != 4:
		# 离散数据处理
		for x_v in set(X[:, chosen_attr]):
			n = Node(-1, -1, x_v)
			root.children.append(n)
			# 不可能Dv empty 要是empty压根不会在set里
			# 选出 X[attr] == x_v的行

			index_x_equal_v = np.where(x_attr_col == x_v)
			X_x_equal_v = X[index_x_equal_v]
			Y_x_equal_v = Y[index_x_equal_v]
			dicision_tree_init_C4_5(X_x_equal_v, Y_x_equal_v, attrs, n, purity_cal)
	else:
		n = Node(-1, -1, point)
		root.children.append(n)
		index1 = np.where(x_attr_col < point)
		index2 = np.where(x_attr_col >= point)
		X1 = X[index1]
		Y1 = Y[index1]
		X2 = X[index2]
		Y2 = Y[index2]
		dicision_tree_init_C4_5(X1, Y1, attrs, n, purity_cal)
		dicision_tree_init_C4_5(X2, Y2, attrs, n, purity_cal)


def dicision_tree_predict_ID3(x, tree_root):
	if tree_root.label != np.pi:
		return tree_root.label

	# 决策
	if tree_root.label == np.pi and tree_root.attr == np.pi:
		print("err!")
		return None

	chose_attr = tree_root.attr
	# 寻找自己应该进入哪个分支
	for child in tree_root.children:
		if child.attr_v == x[chose_attr]:
			return dicision_tree_predict_ID3(x, child)
	return None


def dicision_tree_predict_C4_5(x, tree_root):
	if tree_root.label != np.pi:
		return tree_root.label

	# 决策
	if tree_root.label == np.pi and tree_root.attr == np.pi:
		print("err!")
		return None

	chose_attr = tree_root.attr
	# 寻找自己应该进入哪个分支
	if chose_attr != 4:
		for child in tree_root.children:
			if child.attr_v == x[chose_attr]:
				return dicision_tree_predict_C4_5(x, child)
	else:
		if tree_root.attr_v > x[chose_attr]:
			return dicision_tree_predict_C4_5(x, tree_root.children[0])
		else:
			return dicision_tree_predict_C4_5(x, tree_root.children[1])
	return None


if __name__ == '__main__':
	ans = load_txt1("Watermelon-train1.csv")
	X_train1 = ans[:, 1: -1]
	Y_train1 = ans[:, -1]
	Y_train1.astype(np.int64)
	# print(X_train)
	# print(Y_train)

	test_data = load_txt1("Watermelon-test1.csv")
	X_test1 = test_data[:, 1:-1]
	Y_test1 = test_data[:, -1]

	ans = load_txt2("Watermelon-train2.csv")
	X_train2 = ans[:, 1: -1]
	Y_train2 = ans[:, -1]
	Y_train2.astype(np.int64)
	# print(X_train)
	# print(Y_train)

	test_data = load_txt2("Watermelon-test2.csv")
	X_test2 = test_data[:, 1:-1]
	Y_test2 = test_data[:, -1]

	r1 = Node(-1, -1, -1)
	attrs1 = [0, 1, 2, 3]

	r2 = Node(-1, -1, -1)
	attrs2 = [0, 1, 2, 3, 4]

	dicision_tree_init_ID3(X_train1, Y_train1, attrs1, r1, gain)

	y_predict_ID3 = []
	for i in range(X_test1.shape[0]):
		x = X_test1[i]
		y_p = dicision_tree_predict_ID3(x, r1)
		y_predict_ID3.append(y_p)
	acc1 = accuracy_score(Y_test1, y_predict_ID3)

	dicision_tree_init_C4_5(X_train2, Y_train2, attrs2, r2, gain_ratio)
	y_predict_C4_5 = []
	for i in range(X_test2.shape[0]):
		x = X_test2[i]
		y_p = dicision_tree_predict_C4_5(x, r2)
		y_predict_C4_5.append(y_p)
	acc2 = accuracy_score(Y_test2, y_predict_C4_5)
	acc3 = accuracy_score(Y_test2, y_predict_C4_5)
	print('accuracy_ID3:', acc1)
	print('accuracy_C4.5:', acc2)
	print('accuracy_pre_pruning_C4.5:', acc3)