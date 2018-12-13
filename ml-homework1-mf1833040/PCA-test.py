import Pca
import _1NN
import LoadDataSet
from numpy import *
if __name__ == '__main__':
	#导入数据集
	path_train = 'datasets/sonar-train.txt'
	path_test = 'datasets/sonar-test.txt'
	# path_train = 'datasets/splice-train.txt'
	# path_test = 'datasets/splice-test.txt'
	dataMat_train, tagMat_train = LoadDataSet.loadDataSet(path_train)
	dataMat_test, tagMat_test = LoadDataSet.loadDataSet(path_test)
	#根据维度k来构造投影矩阵W
	k = input("please enter k")
	W = Pca.pca(dataMat_train,int(k))
	#分别投影train和test到k维空间
	dataMat_train_k = dataMat_train * W
	dataMat_test_k = dataMat_test * W
	#使用1NN来计算准确率
	print("当维度为%s时,正确率为" %k, _1NN._1nn(dataMat_train_k, dataMat_test_k, tagMat_train, tagMat_test))




