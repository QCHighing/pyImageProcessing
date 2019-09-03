import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

img = data.astronaut() # 读入图像
img = rgb2gray(img)    # 灰度化

# 圆的参数方程：(220, 100) r=100
t = np.linspace(0, 2*np.pi, 400) # 参数t, [0,2π]
x = 220 + 100*np.cos(t)
y = 100 + 100*np.sin(t)

# 构造初始Snake
init = np.array([x, y]).T # shape=(400, 2)

# Snake模型迭代输出
snake = active_contour(gaussian(img,3), snake=init, alpha=0.1, beta=1, gamma=0.01, w_line=0, w_edge=10)

# 绘图显示
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap="gray")
plt.plot(init[:, 0], init[:, 1], '--r', lw=3)
plt.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
plt.xticks([]), plt.yticks([]), plt.axis("off")
plt.show()
