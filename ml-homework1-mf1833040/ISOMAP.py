from numpy import *
import heapq

# 获取欧氏距离
# data: 要获取欧氏距离的矩阵，大小 m * n
# return：m * m 的矩阵，第 [i, j] 个元素代表 data 中元素 i 到元素 j 的欧氏距离
def get_distance(data):
    data_count = len(data)
    mat_distance = zeros([data_count, data_count], float32)
    for idx in range(data_count):
        for sub_idx in range(data_count):
            mat_distance[idx][sub_idx] = linalg.norm(data[idx] - data[sub_idx])
    return mat_distance

class edge(object):
    def __init__(self, cost, to):
        self.cost = cost
        self.to = to
    def __lt__(self, other):
        return self.cost < other.cost

def dijkstra(dist, graph, src):
    heap = []
    heapq.heappush(heap, edge(0, src))
    while heap:
        p = heapq.heappop(heap)
        v = p.to
        if dist[src][v] < p.cost:
            continue
        for i in range(len(graph[v])):
            if dist[src][graph[v][i].to] > dist[src][v] + graph[v][i].cost:
                dist[src][graph[v][i].to] = dist[src][v] + graph[v][i].cost
                heapq.heappush(heap, edge(dist[src][graph[v][i].to], graph[v][i].to))

# mds 算法的具体实现
# data：需要降维的矩阵
# dims：目标维度
# return：降维后的矩阵
def mds(dist, dims):
    data_count = len(dist)
    if dims > data_count:
        dims = data_count
    val_dist_i_j = 0.0
    vec_dist_i_2 = zeros([data_count], float32)
    vec_dist_j_2 = zeros([data_count], float32)
    mat_b = zeros([data_count, data_count], float32)
    # mat_distance = get_distance(data)
    for idx in range(data_count):
        for sub_idx in range(data_count):
            dist_ij_2 = square(dist[idx][sub_idx])
            val_dist_i_j += dist_ij_2
            vec_dist_i_2[idx] += dist_ij_2
            vec_dist_j_2[sub_idx] += dist_ij_2 / data_count
        vec_dist_i_2[idx] /= data_count
    val_dist_i_j /= square(data_count)
    for idx in range(data_count):
        for sub_idx in range(data_count):
            dist_ij_2 = square(dist[idx][sub_idx])
            mat_b[idx][sub_idx] = -0.5 * (dist_ij_2 - vec_dist_i_2[idx] - vec_dist_j_2[sub_idx] + val_dist_i_j)
    #计算特征值和特征向量，特征值默认已从大到小排好序
    eigVals, eigVecs = linalg.eig(mat_b)
    eigVals_Idx = argsort(eigVals)[:-(dims+1):-1]
    #构建特征值对角矩阵
    eigVals_Diag = diag(maximum(eigVals[eigVals_Idx], 0.0))
    #实现降维
    return matmul(eigVecs[:, eigVals_Idx], sqrt(eigVals_Diag))

# isomap 算法的具体实现
# data：需要降维的矩阵
# dims：目标维度
# k：k 近邻算法中的超参数
# return：降维后的矩阵
def isomap(data, dims, k):
    set_printoptions(threshold=NaN)
    inf = float('inf')
    data_count = len(data)
    if k >= data_count:
        raise ValueError('K的值最大为数据个数 - 1')
    mat_distance = get_distance(data)
    knn_map = ones([data_count, data_count], float32) * inf
    for idx in range(data_count):
        top_k = argpartition(mat_distance[idx], k)[:k + 1]
        knn_map[idx][top_k] = mat_distance[idx][top_k]
        for i in top_k:
            knn_map[i][idx] = knn_map[idx][i]   
    # dijkstra
    # 邻接矩阵存储图
    graph = []
    for i in range(data_count):
        edgelist = []
        for j in range(data_count):
            if knn_map[i][j] != inf:
                edgelist.append(edge(knn_map[i][j], j))
        graph.append(edgelist)
    # dist存储任意两点之间最短距离
    dist = ones([data_count, data_count], float32) * inf
    for idx in range(data_count):
        dist[idx][idx] = 0.0
        dijkstra(dist, graph, idx)
    return mds(dist, dims)