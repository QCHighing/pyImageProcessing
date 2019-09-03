import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def genarateData(N=500):
    X1 = np.random.normal((0, 0), (0.4, 0.4), (N // 2, 2))
    X2 = np.random.normal((1, 1), (0.4, 0.4), (N // 2, 2))
    X = np.vstack([X1, X2])
    Y = np.array([0] * (N // 2) + [1] * (N // 2))
    return X, Y


class myKMeans:
    cluster_centers_ = []
    labels_ = []
    n_iter_ = 0

    def __init__(self, C, G=200):
        self.C = C
        self.G = G

    def fit(self, X):
        m, n = X.shape
        C = self.C
        index = np.random.choice(range(m), size=C, replace=False)
        centers = X[index]
        results = np.hstack([X, np.zeros([m, C + 1])])
        for g in range(self.G):
            # 聚类中心排序
            centers = centers[centers[:, 0].argsort()]
            # 计算欧式距离
            for i in range(C):
                results[:, n + i] = np.sum((X - centers[i])**2, axis=1)
            # 按距离聚类
            results[:, -1] = np.argmin(results[:, n:n + C], axis=1)
            # 更新聚类中心
            pre_centers = centers.copy()
            for i in range(C):
                x_i = (results[results[:, -1] == i])[:, :n]
                n_i = len(x_i)
                centers[i] = np.sum(x_i, axis=0) / n_i
            # 判断是否收敛
            if(((pre_centers - centers) < 1e-10).all()):
                break
        self.cluster_centers_ = centers
        self.labels_ = results[:, -1].astype(np.uint8)
        self.n_iter_ = g + 1


class FCM:
    cluster_centers_ = []
    labels_ = []
    n_iter_ = 0

    def __init__(self, C, G=200):
        self.C = C
        self.G = G

    def calcu_U(self, X, C, U, V):
        N, n = X.shape
        for k in np.arange(N):
            d_Euler = np.zeros(C)  # 存放样本k到C个聚类中心的欧式距离的平方
            for i in range(C):
                # 计算欧式距离
                for j in range(n):
                    d_Euler[i] += (X[k, j] - V[i, j]) ** 2
                # 样本k如果和聚类中心重合，则隶属度为1，此时距离需要令为1
                if(d_Euler[i] == 0):
                    d_Euler[i] = 1e-5
            for i in range(C):
                den = d_Euler[i] * sum(1 / d_Euler)
                U[k, i] = 1 / den
        return U

    def calcu_V(self, X, C, U, V):
        num = (U ** 2).T @ X   # (N,3).T x (N, 4) = (3, 4)
        den = np.sum(U**2, axis=0).reshape(C, 1)
        V = num / den
        V = V[V[:, 0].argsort()]   # 按花萼长度排序，由小到大
        return V

    def defuzzy(self, X, U):          # 根据隶属度最大原则进行聚类
        N, n = X.shape
        Y_predict = np.argmax(U, axis=1)
        return Y_predict

    def fit(self, X):
        m, n = X.shape
        C = self.C
        U = np.zeros([m, C])
        index = np.random.choice(range(m), size=C, replace=False)
        V = X[index].copy()
        for g in range(self.G):
            # 聚类中心排序
            V = V[V[:, 0].argsort()]
            # 计算隶属度矩阵U
            U = self.calcu_U(X, C, U, V)
            # 计算聚类中心
            V_pre = V
            V = self.calcu_V(X, C, U, V)
            # 迭代终止标准
            if((abs(V - V_pre) < 1e-10).all()):
                self.n_iter_ = g + 1
                break
        self.cluster_centers_ = V
        self.labels_ = self.defuzzy(X, U)


def main():
    N = 600
    X, Y = genarateData(N)

    start = time.time()
    cluster1 = myKMeans(C=2)
    cluster1.fit(X)
    end = time.time()
    e1 = abs(Y - cluster1.labels_)
    accuracy1 = 1 - sum(e1) / N
    print(f'my KMeans - 正确率：{accuracy1 : .3f}，迭代次数：{cluster1.n_iter_}，用时：{end - start : .2f} sec')
    print('聚类中心坐标：')
    print(cluster1.cluster_centers_)

    print('\n' + '--' * 20 + '\n')

    start = time.time()
    cluster2 = KMeans(n_clusters=2)
    cluster2.fit(X)
    end = time.time()
    if(np.mean(cluster2.labels_[10]) > 0.5):
        cluster2.labels_ = abs(cluster2.labels_ - 1)
    e2 = abs(Y - cluster2.labels_)
    accuracy2 = 1 - sum(e2) / N
    print(f'sk KMeans - 正确率：{accuracy2 : .3f}，迭代次数：{cluster2.n_iter_}，用时：{end - start : .2f} sec')
    print('聚类中心坐标：')
    print(cluster2.cluster_centers_)

    print('\n' + '--' * 20 + '\n')

    start = time.time()
    cluster3 = FCM(C=2)
    cluster3.fit(X)
    end = time.time()
    e3 = abs(Y - cluster3.labels_)
    accuracy3 = 1 - sum(e3) / N
    print(f'Fuzzy CM - 正确率：{accuracy3 : .3f}，迭代次数：{cluster3.n_iter_}，用时：{end - start : .2f} sec')
    print('聚类中心坐标：')
    print(cluster3.cluster_centers_)


if __name__ == '__main__':
    main()
