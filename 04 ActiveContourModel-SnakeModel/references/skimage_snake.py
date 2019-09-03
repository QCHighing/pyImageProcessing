from warnings import warn
import numpy as np
from scipy.interpolate import RectBivariateSpline
from ..util import img_as_float
from ..filters import sobel

"""
简化后的代码，不可运行，只用于Snake算法逻辑参考
完整代码参见 skimage::segmentation::active_contour
源代码链接：https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/active_contour_model.py
"""

def active_contour(image, snake, alpha=0.01, beta=0.1,
                   w_line=0, w_edge=1, gamma=0.01,
                   bc=None, max_px_move=1.0,
                   max_iterations=2500, convergence=0.1,
                   *,
                   boundary_condition='periodic',
                   coordinates=None):

    img = img_as_float(image)

    # 计算梯度幅值图像
    edge = [sobel(img)]

    # 叠加强度和边缘图像:
    img = w_line * img + w_edge * edge[0]

    x, y = snake[:, 0].astype(np.float), snake[:, 1].astype(np.float)
    n = len(x)
    convergence_order = 10
    xsave = np.empty((convergence_order, n))
    ysave = np.empty((convergence_order, n))

    # 计算迭代式中的五对角循环矩阵A
    a = np.roll(np.eye(n), -1, axis=0) + \
        np.roll(np.eye(n), -1, axis=1) - \
        2 * np.eye(n)  # second order derivative, central difference
    b = np.roll(np.eye(n), -2, axis=0) + \
        np.roll(np.eye(n), -2, axis=1) - \
        4 * np.roll(np.eye(n), -1, axis=0) - \
        4 * np.roll(np.eye(n), -1, axis=1) + \
        6 * np.eye(n)  # fourth order derivative, central difference
    A = -alpha * a + beta * b

    # 利用隐式格式来最小化能量函数只需要计算一次逆矩阵
    inv = np.linalg.inv(A + gamma * np.eye(n))

    # Explicit time stepping for image energy minimization:
    intp = RectBivariateSpline(np.arange(img.shape[1]),
                               np.arange(img.shape[0]),
                               img.T, kx=2, ky=2, s=0)
    for i in range(max_iterations):
        fx = intp(x, y, dx=1, grid=False)
        fy = intp(x, y, dy=1, grid=False)
        xn = inv @ (gamma * x + fx)
        yn = inv @ (gamma * y + fy)

        # Movements are capped to max_px_move per iteration:
        dx = max_px_move * np.tanh(xn - x)
        dy = max_px_move * np.tanh(yn - y)
        x += dx
        y += dy

        # Convergence criteria needs to compare to a number of previous
        # configurations since oscillations can occur.
        j = i % (convergence_order + 1)
        if j < convergence_order:
            xsave[j, :] = x
            ysave[j, :] = y
        else:
            dist = np.min(np.max(np.abs(xsave - x[None, :]) +
                                 np.abs(ysave - y[None, :]), 1))
            if dist < convergence:
                break
