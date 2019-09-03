import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def calcu_U(X, C, U, V):
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


def calcu_V(X, C, U, V):
    num = (U ** 2).T @ X   # (N,3).T x (N, 4) = (3, 4)
    den = np.sum(U**2, axis=0).reshape(C, 1)
    V = num / den
    V = V[V[:, 0].argsort()]   # 按花萼长度排序，由小到大
    return V


def fcm(X, C, U, V, G=100):
    for g in range(G):
        # 计算隶属度矩阵U
        U = calcu_U(X, C, U, V)
        # 计算聚类中心
        V_pre = V
        V = calcu_V(X, C, U, V)
        # 迭代终止标准
        if((abs(V - V_pre) < 1e-5).all()):
            print(f"迭代次数：{g}")
            break
    return V


def predict(X, C, U, V):    # 按照欧式距离最小原则进行聚类
    N, n = X.shape
    Y_predict = np.zeros([N, 1])
    for k in range(N):
        d_Euler = np.zeros(C)
        for i in range(C):
            for j in range(n):
                d_Euler[i] += (X[k, j] - V[i, j]) ** 2
        Y_predict[k, 0] = np.argmin(d_Euler)
    return Y_predict


def defuzzy(X, U):          # 根据隶属度最大原则进行聚类
    N, n = X.shape
    Y_predict = np.argmax(U, axis=1).reshape(N, 1)
    return Y_predict


def display(X, Y, Y_predict):
    data = np.hstack([X, Y])
    c1 = data[data[:, -1] == 0]
    c2 = data[data[:, -1] == 1]
    c3 = data[data[:, -1] == 2]
    data_predict = np.hstack([X, Y_predict])
    c1_predict = data_predict[data_predict[:, -1] == 0]
    c2_predict = data_predict[data_predict[:, -1] == 1]
    c3_predict = data_predict[data_predict[:, -1] == 2]
    
    plt.figure('Iris数据集-FCM聚类示意图', figsize=(10, 5))
    plt.subplot(121), plt.title('原分类')
    plt.scatter(c1[:, 0], c1[:, 1], c="r", marker='o', label='Y_1')
    plt.scatter(c2[:, 0], c2[:, 1], c="g", marker='x', label='Y_2')
    plt.scatter(c3[:, 0], c3[:, 1], c="b", marker='+', label='Y_3')
    plt.xlabel('花萼长度 / cm'), plt.ylabel('花萼宽度 / cm'), plt.legend(loc=2)
    plt.subplot(122), plt.title('FCM聚类')
    plt.scatter(c1_predict[:, 0], c1_predict[:, 1], c="r", marker='o', label='Y_1')
    plt.scatter(c2_predict[:, 0], c2_predict[:, 1], c="g", marker='x', label='Y_2')
    plt.scatter(c3_predict[:, 0], c3_predict[:, 1], c="b", marker='+', label='Y_3')
    plt.xlabel('花萼长度 / cm'), plt.ylabel('花萼宽度 / cm'), plt.legend(loc=2)
    plt.show()


def main():
    iris_data = pd.read_csv('iris.data', names=['A', 'B', 'C', 'D', 'Y'], header=None)
    iris_data_new = iris_data.replace({'Y': {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}})
    iris_data_np = np.array(iris_data_new)

    np.random.shuffle(iris_data_np)     # 样本随机排序
    X = iris_data_np[:, :-1]
    Y = iris_data_np[:, -1:]
    N, n = X.shape                      # 样本数N,特征数n
    C = 3                               # 聚类中心数
    U = np.zeros([N, C])                # 隶属度矩阵U
    V = X[:C].copy()                    # 随机选择C个样本初始化聚类中心V
    G = 50                              # 最大迭代次数

    start = time.time()
    V = fcm(X, C, U, V, G)              # FCM算法迭代计算聚类中心
    end = time.time()

    Y_predict = predict(X, C, U, V)     # 按欧式距离聚类
    Y_predict = defuzzy(X, U)           # 解模糊

    e = abs(Y - Y_predict)
    e[e > 1] = 1
    print(f"Accuracy = {(1 - np.sum(e) / N) * 100 : .2f} %")
    print(f"FCM time = {end - start : .2f} sec")

    display(X, Y, Y_predict)

    return 0


if __name__ == '__main__':
    main()
