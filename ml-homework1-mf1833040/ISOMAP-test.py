import ISOMAP_Dijkstra
import _1NN
from numpy import *
import time

if __name__ == '__main__':
	timestamp = time.time()
	#导入数据集
	path_train = 'datasets/sonar-train.txt'
	path_test = 'datasets/sonar-test.txt'
	# path_train = 'datasets/splice-train.txt'
	# path_test = 'datasets/splice-test.txt'
	dataMat_train, tagMat_train = loadDataSet.loadDataSet(path_train)
	dataMat_test, tagMat_test = loadDataSet.loadDataSet(path_test)

	#根据
	k = input("please enter k：")
	d = input("please enter x-NN：")
	dataMat_train_k = ISOMAP_Dijkstra.isomap(vstack([dataMat_train, dataMat_test]), int(k), int(d))
	#使用1NN来计算准确率
	print("(ISOMAP on sonar)当维度为%s时,正确率为" %k, _1NN._1nn(dataMat_train_k[range(0, len(dataMat_train)), :], dataMat_train_k[range(len(dataMat_train),len(dataMat_train_k)), :], mat(tagMat_train), mat(tagMat_test)))
	print("time used of ISOMAP test on sonar：", time.time() - timestamp, 's')