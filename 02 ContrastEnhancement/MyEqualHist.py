import numpy as np
import cv2 as cv


def myEH(img):
    imgray = img.copy()
    # 灰度级频率统计
    rows, cols = imgray.shape
    bins = np.zeros(256, dtype=np.uint16)
    for r in range(rows):
        for c in range(cols):
            v = imgray[r, c]
            bins[v] += 1
    # 计算灰度级概率分布
    pros = np.zeros(256)
    total = rows * cols
    for i in range(0, 256):
        pros[i] = bins[i] / total
    # 计算累加概率分布
    for i in range(1, 256):
        pros[i] = pros[i - 1] + pros[i]
    # 映射新灰度
    for r in range(rows):
        for c in range(cols):
            v = imgray[r, c]
            imgray[r, c] = round(pros[v] * 255)  # 四舍五入取整
    return imgray


def main():
    # 读入源图像
    imsrc = cv.imread('../img/Lenna.png')

    # 灰度处理
    imgray = cv.cvtColor(imsrc, cv.COLOR_BGR2GRAY)

    # 调用openCV直方图均衡函数
    imEH = cv.equalizeHist(imgray)

    # 调用自写直方图均衡函数
    imMEH = myEH(imgray)

    # 显示图像
    dst = np.hstack((imgray, imEH, imMEH))  # 并排存放两张图像
    cv.imshow('myEH', dst)
    cv.imwrite('../img/comparison.png', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
