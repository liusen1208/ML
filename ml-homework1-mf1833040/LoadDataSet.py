from numpy import *

def loadDataSet(filename):
    # 打开文本文件
    fr = open(filename)
    # 对文本中每一行的特征分隔开来，存入列表中，作为列表的某一行
    # 行中的每一列对应各个分隔开的特征
    L = fr.readlines()
    #分别读取Data和Tag放入list中
    stringData = [line.strip().split(",")[:-1] for line in L]
    stringTag = [line.strip().split(",")[-1:] for line in L]
    # 利用map()函数，将列表中每一行的数据值映射为float，int型
    Data = [list(map(float,line))for line in stringData]
    Tag = [list(map(int,line))for line in stringTag]
    # 将Data和Tag数据值的列表转化为矩阵返回
    return mat(Data),mat(Tag)