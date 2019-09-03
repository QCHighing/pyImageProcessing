import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from time import time


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def calcu_U(X, C, U, V):
    N = len(X)
    for k in np.arange(N):
        d_square = np.zeros(C)
        for i in range(C):
            d_square[i] = (X[k, 0] - V[i, 0]) ** 2
            if(d_square[i] == 0):
                d_square[i] = 1e-5
        for i in range(C):
            den = d_square[i] * sum(1 / d_square)
            U[k, i] = 1 / den
    return U


def calcu_V(X, C, U, V):
    num = (U ** 2).T @ X
    den = np.sum(U**2, axis=0).reshape(C, 1)
    V = num / den
    V = V[V[:, 0].argsort()]
    return V


def fcm(X, C, U, V, G=100):
    V_pre = V.copy()
    for g in range(G):
        # 计算隶属度矩阵U
        U = calcu_U(X, C, U, V)
        # 计算聚类中心
        V_pre = V
        V = calcu_V(X, C, U, V)
        if((abs(V_pre - V) < 1e-5).all()):
            print(f"迭代次数：{g}")
            break
    V = np.around(V)
    V[V > 255] = 255
    return V.astype(np.uint8)


def defuzzy(src, C, U, V):          # 根据隶属度最大原则进行聚类
    M, N = src.shape
    X = src.ravel().reshape(M * N, 1)
    u = np.argmax(U, axis=1).reshape(M * N, 1)
    X_u = np.hstack([X, u])
    for c in range(C):
        X_u[X_u[:, 1] == c] = V[c]
    imclust = X_u[:, 1].reshape(M, N)
    return imclust


def display(src, dst):
    plt.figure("OpenCV Logo")
    plt.subplot(121), plt.imshow(src, cmap='gray'), plt.title('Source Image'), plt.axis("off")
    plt.subplot(122), plt.imshow(dst, cmap='gray'), plt.title('Clustered Image'), plt.axis("off")
    plt.show()


def main():
    src = cv.imread("opencv_logo.png", 0)   # size = (555, 599)
    res = cv.resize(src, (256, 256))

    plt.figure("Image Histogram")
    plt.hist(res.ravel(), bins=256, range=[0, 256])
    plt.title('Image Histogram'), plt.xlabel('灰度值'), plt.ylabel('像素个数')
    # plt.show()  禁止在此处输出

    M, N = res.shape
    X = res.ravel().reshape(M * N, 1)
    C = 4
    U = np.zeros([M * N, C])
    V = np.random.randint(0, 255, [C, 1])
    G = 50

    start = time()
    V = fcm(X, C, U, V, G)              # FCM算法迭代计算聚类中心
    end = time()
    print(f"FCM time = {end-start : .2f} sec")
    print(f"聚类中心灰度值：{V.T}")

    dst = defuzzy(res, C, U, V)

    display(res, dst)


if __name__ == '__main__':
    main()
