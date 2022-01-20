import numpy as np
import matplotlib.pyplot as plt
import time


def data_loading(file):
	fr = open(file, 'r')
	feature = fr.readline()
	# print(feature)
	tmp_data = fr.readlines()
	fr.close()
	data = []
	for line in tmp_data:
		ls = line.strip().split(',')
		line_s = [float(ls[i]) for i in range(np.size(ls))]
		data.append(line_s)
	# print(np.shape(data))
	return np.array(data)


## normalize
def norm(x):
	x = (x - np.mean(x, 0)) / (np.max(x, 0) - np.min(x, 0))
	return x


def predict(alpha, beta, x):
	arr = alpha * x
	return np.sum(arr) + beta


def gradient_descent(x, y, alpha, beta, learn_rate):
	# gradient_arr是整个 alpha 偏导数数组
	gradient_arr = np.zeros((1, x.shape[1]))
	gradient_beta = 0
	mean_s_err = 0
	for line in range(x.shape[0]):
		xline = x[line, :]
		yline = y[line]
		# err = y - (alpha X + beta)
		err = yline - predict(alpha, beta, xline)
		gradient_arr += err * xline
		gradient_beta += err
		mean_s_err += err ** 2

	# arr 是 alpha vector的梯度vec， alpha0 是 arr[0]
	gradient_arr = gradient_arr * 2 / x.shape[0]
	gradient_beta = gradient_beta * 2 / x.shape[0]
	mean_s_err /= x.shape[0]

	alpha += np.reshape(gradient_arr, alpha.shape) * learn_rate
	beta += gradient_beta * learn_rate
	return alpha, beta, mean_s_err


def gradient_descent_random(x, y, alpha, beta, learn_rate):
	randomId = int(np.random.random_sample() * x.shape[0])
	x = x[randomId, :]
	y = y[randomId]
	gradient_arr = np.zeros(x.shape[0])
	gradient_beta = 0
	err = y - predict(alpha, beta, x)
	gradient_arr += err * x
	gradient_beta += err

	# arr 是 alpha vector的梯度vec， alpha0 是 arr[0]
	gradient_arr = gradient_arr * 2
	gradient_beta = gradient_beta * 2

	alpha += np.reshape(gradient_arr, alpha.shape) * learn_rate
	beta += gradient_beta * learn_rate

	return alpha, beta


def gradient_descent_random_L2(x, y, alpha, beta, learn_rate, L2_lambda):
	randomId = int(np.random.random_sample() * x.shape[0])
	x = x[randomId, :]
	y = y[randomId]
	gradient_arr = np.zeros(x.shape[0])
	gradient_beta = 0
	err = y - predict(alpha, beta, x)
	gradient_arr += err * x
	gradient_arr -= alpha * L2_lambda
	gradient_beta += err

	# arr 是 alpha vector的梯度vec， alpha0 是 arr[0]
	gradient_arr = gradient_arr * 2
	gradient_beta = gradient_beta * 2

	alpha += np.reshape(gradient_arr, alpha.shape) * learn_rate
	beta += gradient_beta * learn_rate

	return alpha, beta


def train_model(x, y, learn_rate, loop_times, method):
	# random init alpha, beta
	alpha = np.random.random_sample(x.shape[1])
	beta = np.random.random_sample()
	mean_s_err = 0
	if method == "SGD":
		for i in range(loop_times):
			alpha, beta = gradient_descent_random(x, y, alpha, beta, learn_rate)
		return alpha, beta
	elif method == "SGD_L2":
		for i in range(loop_times):
			alpha, beta = gradient_descent_random_L2(x, y, alpha, beta, learn_rate, 1)
		return alpha, beta
	else:
		for i in range(loop_times):
			alpha, beta, mean_s_err = gradient_descent(x, y, alpha, beta, learn_rate)
		return alpha, beta, mean_s_err


if __name__ == "__main__":
	Data = data_loading('winequality-white.csv')
	X, Y = Data[:, :-1], Data[:, -1]
	# print(np.shape(X))
	X = norm(X)

	# learn_rate = input('Input learn_rate: ')
	# learn_rate = [0.05, 0.07, 0.09, 0.1]
	learn_rate = 0.1
	mean_s_err_vec = []
	mean_s_err = 0
	# t0 = time.process_time()
	fig, ax = plt.subplots(figsize=(16, 9))
	x_loop = [i for i in range(5, 100, 10)]  # 迭代次数
	# for rate in learn_rate:
	# 	mean_s_err = 0
	# 	mean_s_err_vec = []
	# for loop in x_loop:
	# 	alpha, beta, mean_s_err = train_model(X, Y, float(learn_rate), loop, "BGD")
	# 	mean_s_err_vec.append(np.sqrt(mean_s_err))
	# 	print(mean_s_err)
	# 	# ax.plot(x_loop, mean_s_err_vec, "-", label="learn_rate=" + str(rate))
	# ax.plot(x_loop, mean_s_err_vec, "-", label="BGD")
	# mean_s_err = 0
	# mean_s_err_vec=[]
	for loop in x_loop:
		for i in range(X.shape[0]):
			# 留一法
			X_test = X[i, :]
			Y_test = Y[i]
			X_train = np.delete(X, i, axis=0)
			Y_train = np.delete(Y, i)

			alpha, beta = train_model(X_train, Y_train, float(learn_rate), loop, "SGD")
			err = Y_test - predict(alpha, beta, X_test)
			mean_s_err += err ** 2
		mean_s_err /= X.shape[0]
		mean_s_err_vec.append(np.sqrt(mean_s_err))
	ax.plot(x_loop, mean_s_err_vec, "-", label="SGD")
	mean_s_err = 0
	mean_s_err_vec=[]
	for loop in x_loop:
		for i in range(X.shape[0]):
			# 留一法
			X_test = X[i, :]
			Y_test = Y[i]
			X_train = np.delete(X, i, axis=0)
			Y_train = np.delete(Y, i)

			alpha, beta = train_model(X_train, Y_train, float(learn_rate), loop, "SGD_L2")
			err = Y_test - predict(alpha, beta, X_test)
			mean_s_err += err ** 2
		mean_s_err /= X.shape[0]
		mean_s_err_vec.append(np.sqrt(mean_s_err))
	ax.plot(x_loop, mean_s_err_vec, "-", label="SGD_L2")
	# print('process_time:', time.process_time()-t0)
	print('loop:', x_loop)
	# print('RMSE:', mean_s_err_vec)
	ax.set_xlabel("loop time")
	ax.set_ylabel("RMSE")
	ax.legend(loc="best")
	ax.set_title("RMSE of SGD and SGD_L2")
	# plt.plot(x_loop, mean_s_err_vec, "-", label="learn_rate=" + learn_rate)
	# plt.xlabel("loop time")
	# plt.ylabel("RMSE")
	# plt.title("RMSE of regression model")
	# plt.legend(loc="best")
	# plt.savefig("fig.png")
	plt.show()
