import numpy as np

MAX_NUM = 1e3


# method
def singleLinkage(X, init):
    if init:
        return np.min(X)
    return np.min(X, axis=0)


def completeLinkage(X, init):
    if init:
        return np.max(X)
    return np.max(X, axis=0)


def averageLinkage(X, init):
    if init:
        return np.average(X)
    return np.average(X, axis=0)


class AgglomerativeClustering:
    def __init__(self):
        self.steps = []

    def fit(self, datas, method):
        self.dataCnt = datas.shape[0]
        # 预处理各点之间的距离
        allDist = np.zeros((self.dataCnt, self.dataCnt))
        for i in range(self.dataCnt):
            for j in range(i):
                allDist[i][j] = allDist[j][i] = np.sum((datas[i] - datas[j]) ** 2)
        setList, clusterCount = [[i] for i in range(self.dataCnt)], self.dataCnt
        print("calculate distance finish!")

        # 每个聚类之间的相似度矩阵
        clusterDist = np.zeros((self.dataCnt, self.dataCnt)) + MAX_NUM
        for i in range(clusterCount):
            for j in range(i + 1, clusterCount):
                clusterDist[i][j] = clusterDist[j][i] = allDist[i][j]
        print("calculate cluster distance finish!")

        while clusterCount != 1:
            # 最相似的两个聚类
            res = np.argmin(clusterDist)
            dest, src = int(res / clusterCount), res % clusterCount
            # steps modify
            self.steps.append((setList[dest][0], setList[src][0]))
            # cluster distance modify
            modify = method(clusterDist[[dest, src]], False)
            clusterDist[dest] = modify
            clusterDist[:, dest] = modify
            clusterDist = np.delete(clusterDist, src, axis=0)
            clusterDist = np.delete(clusterDist, src, axis=1)
            clusterDist[dest][dest] = MAX_NUM
            # cluster modify
            setList[dest] = setList[dest] + setList[src]
            del setList[src]
            clusterCount -= 1
            if (self.dataCnt - clusterCount) % (self.dataCnt / 20) == 0:
                print(clusterCount, " clusters left.")

        print("cluster finish !")

    def label(self, k):
        root = list(range(self.dataCnt))

        def find_root(n):
            if root[root[n]] == root[n]:
                return root[n]
            root[n] = find_root(root[n])
            return root[n]

        for i in range(self.dataCnt - k):  # 根据steps记录产生非连通图
            src, dest = self.steps[i]
            root[find_root(dest)] = find_root(src)
        cluster, clusterNum = [0 for i in range(self.dataCnt)], 0
        for i in range(self.dataCnt):  # 将根节点标注为新的cluster
            if i == root[i]:  # i is root
                clusterNum += 1
                cluster[i] = clusterNum
        for i in range(self.dataCnt):  # 将非根节点标注为根节点的cluster
            if i != root[i]:  # i is not root
                cluster[i] = cluster[find_root(i)]
        return cluster
