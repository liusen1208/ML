from numpy import *

    #调用svd计算出投影矩阵W
def svd(dataMat, k):
    (U, S, VT) = linalg.svd(dataMat)
    topk = argsort(S)[:-(k+1):-1]
    new_VT = VT[topk,:]
    W = new_VT.T
    return W




