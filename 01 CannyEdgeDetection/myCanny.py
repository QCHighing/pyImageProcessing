import numpy as np
import cv2 as cv


def dispImg(imgs):
    for name, img in imgs.items():
        cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def calcuGradient(img, imdx, imdy, immod, imdir):
    row, col = img.shape
    for r in range(1, row - 1):   # 去掉最外围的1像素，从1索引到row-1, col-1
        for c in range(1, col - 1):
            imdx[r, c] = dx = img[r + 1, c - 1] + 2 * img[r + 1, c] + img[r + 1, c + 1] - img[r - 1, c - 1] - 2 * img[r - 1, c] - img[r - 1, c + 1]
            imdy[r, c] = dy = img[r - 1, c + 1] + 2 * img[r, c + 1] + img[r + 1, c + 1] - img[r - 1, c - 1] - 2 * img[r, c - 1] - img[r + 1, c - 1]
            immod[r, c] = np.sqrt(dx ** 2 + dy ** 2)
            theta = np.arctan(dy / dx) * 180 / np.pi if dx else 0  # 梯度方向角：-90~90度
            # imang[r, c] = theta
            if 0 <= theta < 45:
                imdir[r, c] = 0
            elif 45 <= theta <= 90:
                imdir[r, c] = 1
            elif -90 <= theta < -45:
                imdir[r, c] = 2
            else:
                imdir[r, c] = 3
    # 规范化，注意防止溢出
    maxmod = np.max(immod)
    # print(maxmod, type(immod), immod.dtype)
    if maxmod:
        immod = immod * 255 // maxmod
    # 统计取值范围
    # plt.hist(imdx.ravel(), bins=72, range=[-300, 300])  # 参数：一维数组、 bin的个数、 取值范围
    # plt.show()


def NMS(imdx, imdy, immod, imdir, imnms):   # Non-Maximum Suppression 非极大值抑制
    row, col = immod.shape
    for r in range(2, row - 2):
        for c in range(2, col - 2):
            E = immod[r, c + 1]
            W = immod[r, c - 1]
            S = immod[r + 1, c]
            N = immod[r - 1, c]
            NE = immod[r - 1, c + 1]
            NW = immod[r - 1, c - 1]
            SW = immod[r + 1, c - 1]
            SE = immod[r + 1, c + 1]
            gradm = immod[r, c]     # 中心像素的梯度模值
            if imdir[r, c] == 0:
                tanθ = abs(imdy[r, c] / imdx[r, c]) if imdx[r, c] else 0
                gradm1 = (1 - tanθ) * S + tanθ * SE
                gradm2 = (1 - tanθ) * N + tanθ * NW
            elif imdir[r, c] == 1:
                tanθ = abs(imdx[r, c] / imdy[r, c]) if imdy[r, c] else 0
                gradm1 = (1 - tanθ) * E + tanθ * SE
                gradm2 = (1 - tanθ) * W + tanθ * NW
            elif imdir[r, c] == 2:
                tanθ = abs(imdx[r, c] / imdy[r, c]) if imdy[r, c] else 0
                gradm1 = (1 - tanθ) * E + tanθ * NE
                gradm2 = (1 - tanθ) * W + tanθ * SW
            elif imdir[r, c] == 3:
                tanθ = abs(imdy[r, c] / imdx[r, c]) if imdx[r, c] else 0
                gradm1 = (1 - tanθ) * N + tanθ * NE
                gradm2 = (1 - tanθ) * S + tanθ * SW
            else:
                gradm1 = gradm2 = 100
            # 极值比较
            if gradm >= int(gradm1) and gradm >= int(gradm2):
                imnms[r, c] = gradm     # 极大值，保留
            else:
                imnms[r, c] = 0         # 非极大值，抑制


def getCannyThres(img, edgeRatio=0.1, thresRatio=0.4):
    row, col = img.shape
    index = int((1 - edgeRatio) * row * col)
    highThresh = np.sort(img.ravel())[index]
    lowThresh = int(thresRatio * highThresh)
    return lowThresh, highThresh


def DTD(imnms, lowThresh, highThresh, imedge):  # Double Thresholds Detection
    row, col = imnms.shape
    imthres = np.zeros((row, col), dtype=np.uint8)    # 强弱边缘标记矩阵，与源图像同尺寸
    for r in range(0, row):
        for c in range(0, col):
            gradm = imnms[r, c]
            if gradm >= highThresh:
                imthres[r, c] = highThresh   # 强边缘标记
                imedge[r, c] = 255
            elif gradm >= lowThresh:
                imthres[r, c] = lowThresh    # 弱边缘标记
            else:
                imthres[r, c] = 0            # 非边缘抑制
    for r in range(1, row - 1):
        for c in range(1, col - 1):
            if imthres[r, c] == lowThresh:
                neighbors = [imthres[r - 1, c - 1], imthres[r - 1, c], imthres[r - 1, c + 1], imthres[r, c - 1], imthres[r, c + 1], imthres[r + 1, c - 1], imthres[r + 1, c], imthres[r + 1, c + 1]]
                if highThresh in neighbors:
                    imthres[r, c] = highThresh
                    imedge[r, c] = 255      # 与强边缘连接的弱边缘也是边缘，抑制孤立噪点


def main():
    imgs = {}   # 空字典，存放过程图像

    # 读入图像
    imsrc = cv.imread('G:/PythonProjects/MyCV/Pictures/sudoku.jpg')
    # imgs['source image'] = imsrc

    # 灰度化
    imgray = cv.cvtColor(imsrc, cv.COLOR_BGR2GRAY)
    # imgs['gray image'] = imgray

    # 高斯滤波
    imblur = cv.GaussianBlur(imgray, (5, 5), 0)
    # imgs['blur image'] = imblur

    # 设置复制边框，各边增宽2个像素
    imborder = cv.copyMakeBorder(imblur, 2, 2, 2, 2, cv.BORDER_REPLICATE)

    # 求梯度
    row, col = imborder.shape
    imdx = np.zeros((row, col), dtype=np.int16)       # x 方向，即竖直方向的梯度
    imdy = np.zeros((row, col), dtype=np.int16)       # y 方向，即水平方向的梯度
    immod = np.zeros((row, col), dtype=np.uint16)     # 梯度幅值图像，各边扩宽1像素，便于NMS处理
    imdir = np.zeros((row, col), dtype=np.uint8)      # 梯度方向区域矩阵，各边扩宽1像素，便于NMS处理
    calcuGradient(imborder, imdx, imdy, immod, imdir)
    immod = np.uint8(immod)
    # imgs['gradient module image.jpg'] = immod

    # 非极大值抑制
    imnms = np.zeros((row, col), dtype=np.uint8)      # 抑制后的梯度幅值图像
    NMS(imdx, imdy, immod, imdir, imnms)
    imgs['NMS.jpg'] = imnms

    # 计算双阈值
    lowThresh, highThresh = getCannyThres(imnms, edgeRatio=0.1, thresRatio=0.4)
    print(lowThresh, highThresh)

    # 双阀值算法连接边缘
    imedge = np.zeros((row, col), dtype=np.uint8)     # 最终的边缘连接图像
    DTD(imnms, lowThresh, highThresh, imedge)
    imgs['edge.jpg'] = imedge

    # 输出图像
    dispImg(imgs)


if __name__ == '__main__':
    main()
