import numpy as np


# 随机生成 5x5 的两位数矩阵
A = np.random.randint(10, 100, (5, 5))

# 在 [100, 200) 之间随机挑选3个数，返回一个数组
B = np.random.choice(range(100, 200), 3)

# 产生[0,1)的随机数
n = np.random.rand()
