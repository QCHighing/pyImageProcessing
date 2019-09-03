import numpy as np
import cv2 as cv


def remapping(img, levels, chrom):
    immap = img.copy()
    rows, cols = immap.shape
    # 映射新灰度
    for r in range(rows):
        for c in range(cols):
            v = immap[r, c]
            index = levels.index(v)
            immap[r, c] = chrom[index]
    return immap


def cafitness(img):
    # Sobel 边缘检测
    dx = cv.Sobel(img, cv.CV_16U, 1, 0)
    dy = cv.Sobel(img, cv.CV_16U, 0, 1)
    # mv = np.sqrt(dx ** 2 + dy ** 2)
    # imedge = cv.convertScaleAbs(mv)
    absX = cv.convertScaleAbs(dx)
    absY = cv.convertScaleAbs(dy)
    imedge = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    # 计算边缘总强度
    sumE = np.sum(imedge)
    # 检测边缘数目
    _, contours, _ = cv.findContours(imedge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    nEdge = len(contours)
    # 计算适应度
    fitness = np.log(np.log(sumE)) * nEdge
    print(nEdge, '\n', sumE, '\n', fitness, '\n\n')
    return fitness


# 产生一组无重复的升序排列的n个属于[a, b]的随机整数，包含端点a,b
# def diffrand(a, b, n):
#     X = [a]
#     while len(X) < n - 1:
#         x = np.random.randint(a, b)   # 左闭右开
#         if x not in X:
#             X.append(x)
#     X.append(b)
#     return sorted(X)

# 产生一组升序排列的n个属于[a, b]的随机整数，包含端点a,b
def diffrand(a, b, n):
    ints = [a] + np.random.choice(range(a, b + 1), n - 2).tolist() + [b]
    ints.sort()
    return ints


def main():
    # 读入源图像，256 x 256
    imsrc = cv.imread('../img/5236.jpg')

    # 灰度处理
    imgray = cv.cvtColor(imsrc, cv.COLOR_BGR2GRAY)

    # 灰度级频率统计
    rows, cols = imgray.shape
    bins = np.zeros(256, dtype=np.uint16)
    for r in range(rows):
        for c in range(cols):
            v = imgray[r, c]
            bins[v] += 1

    # 计算原始灰度级和灰度级总数 N
    bins = bins.tolist()
    # levels = [bins.index(x) for x in bins if x]  # 剔除零频灰度，注：index函数在数据量大时频繁出错
    levels = []
    for i, x in enumerate(bins):
        if x:
            levels.append(i)
    N = len(levels)

    # 构建初始种群：population number = 10
    pn = 10
    chroms = np.zeros((pn, N), dtype=np.uint8)
    for i in range(pn):
        chroms[i] = diffrand(0, 255, N)

    # 进入迭代 100 次
    lastmaxf = 0
    zerocnt = 0
    for ga in range(10):
        # 计算适应度，选出适应度最优的两个个体
        fitness = []
        for i in range(pn):
            # 增强图像
            immap = remapping(imgray, levels, chroms[i])
            # 计算适应度
            f = cafitness(immap)
            # 保存增强图像
            cv.imwrite(f'../pro/{ga}{i}_f={f}.jpg', immap)
            fitness.append(f)
        print(fitness)
        temp = sorted(fitness)   # 从小到大排，不改变原数组
        fst = fitness.index(temp[-1])
        snd = fitness.index(temp[-2])
        print(fst, snd)

        # 判断迭代条件是否满足
        diff = fitness[fst] - lastmaxf
        if not diff:
            zerocnt += 1
            if zerocnt >= 3:
                break
        elif diff < lastmaxf * 0.02:
            break
        else:
            zerocnt = 0
            lastmaxf = fitness[fst]

        # 进行选择
        temp1, temp2 = chroms[fst].copy(), chroms[snd].copy()
        chroms[fst], chroms[snd] = chroms[0].copy(), chroms[1].copy()
        chroms[0], chroms[1] = temp1.copy(), temp2.copy()
        np.random.shuffle(chroms[2:])

        # 进行交叉
        for i in range(2, 10, 2):
            temp = np.random.choice(range(1, N - 1), 2)
            a, b = min(temp), max(temp)
            chroms[i][a:b + 1], chroms[i + 1][a:b + 1] = chroms[i + 1][a:b + 1].copy(), chroms[i][a:b + 1].copy()
            sorted(chroms[i])
            sorted(chroms[i + 1])

        # 进行突变
        pm = 0.1
        for i in range(2, 10):
            if np.random.rand() < pm:
                index = np.random.choice(range(1, N - 1))
                a = chroms[i][index - 1]
                b = chroms[i][index + 1]
                chroms[i][index] = np.random.choice((a + 1, b))   # 在a,b之中选

    # 显示处理结果
    # dst = np.hstack((imgray, immap))  # 并排存放两张图像
    # cv.imshow('myEH', dst)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


if __name__ == '__main__':
    main()
