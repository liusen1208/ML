from numpy import *
	#1nn算法
def _1nn(dataMat_train, dataMat_test, tag_train, tag_test):
	#记录行数
	N_train = dataMat_train.shape[0]
	N_test = dataMat_test.shape[0]
	#初始化正确数
	N_correct = 0
	#1NN，计算正确数
	#遍历测试集
	for i in range(N_test):
	#取测试样本
		Sample_test = mat(dataMat_test[i, :])
	#遍历训练集
		for j in range(N_train):
	#取测试样本
			Sample_train = mat(dataMat_train[j, :])
	#通过循环比较来记录dis最小值以及此时的index
			if j == 0:
				min_dis = linalg.norm(Sample_train - Sample_test)
				index = 0
			else:
				dis = linalg.norm(Sample_train - Sample_test)
				if dis < min_dis:
					min_dis = dis
					index = j
	#如果命中则N_correct+1
		if(tag_test[i] == tag_train[index]):
			N_correct = N_correct + 1
	#返回正确率
	return N_correct/N_test
