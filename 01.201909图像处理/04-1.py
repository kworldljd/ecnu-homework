##Butterworth 算法，参考：https://blog.csdn.net/cjsh_123456/article/details/79342300
#matlab转python进行试验。

import numpy as np

def Bfilter(image_in, D0, N):
    # % Butterworth滤波器，在频率域进行滤波
    # % 输入为需要进行滤波的灰度图像，Butterworth滤波器的截止频率D0，阶数N
    # % 输出为滤波之后的灰度图像
    print("BFilter :",D0,N)
    (m, n) = image_in.shape
    P = 2 * m
    Q = 2 * n

    fp = np.zeros((P, Q))
    # % 对图像填充0, 并且乘以(-1) ^ (x + y)
    # 以移到变换中心
    for i in np.arange(1, m): #1: m
        for j in np.arange(1, n): # 1: n
            fp[i, j] = image_in[i, j] * ((-1) ** (i + j))

#    % 对填充后的图像进行傅里叶变换
    F1 = np.fft.fft2(fp)

    #% 生成Butterworth滤波函数，中心在(m + 1, n + 1)
    Bw = np.zeros((P, Q))
    a = D0 ^ (2 * N)
    for u in np.arange(1, P): # 1: P
        for v in np.arange(1, Q):
            temp = (u - (m + 1.0)) ** 2 + (v - (n + 1.0)) ** 2
            Bw[u, v] = 1 / (1 + (temp ** N) / a)

   # % 进行滤波
    G = F1 * Bw

    #% 反傅里叶变换
    gp = np.fft.ifft2(G)

    #% 处理得到的图像
    image_out = np.zeros((m, n), dtype='uint8')
    gp = np.real(gp)
    g = np.zeros((m, n))
    for i in np.arange(1, m): #1: m
        for j in np.arange(1, n): # 1: n
            g[i, j] = gp[i, j] * ((-1) ** (i + j))

    mmax = np.max(g)
    mmin = np.min(g)
    range = mmax - mmin
    for i in np.arange(1, m): #1: m
        for j in np.arange(1, n): # 1: n
            image_out[i, j] = np.uint8(255 * (g[i, j] - mmin) / range)

    return image_out

import cv2 as cv
import matplotlib.pyplot as plot

image1 = cv.imread("./source/fonttest.png",cv.IMREAD_GRAYSCALE)
image2 = Bfilter(image1, 10, 2)
image3 = Bfilter(image1, 30, 2)
image4 = Bfilter(image1, 60, 2)
image5 = Bfilter(image1, 160, 2)
image6 = Bfilter(image1, 460, 2)

#% 显示图像
figure, axes = plot.subplots(2, 3)

axes[0, 0].imshow(image1, cmap="gray"), axes[0, 0].set_title("原图")
axes[0, 1].imshow(image2, cmap="gray"), axes[0, 1].set_title("D0 = 10,  n = 2")
axes[0, 2].imshow(image3, cmap="gray"), axes[0, 2].set_title("D0 = 30,  n = 2")
axes[1, 0].imshow(image4, cmap="gray"), axes[1, 0].set_title("D0 = 60,  n = 2")
axes[1, 1].imshow(image5, cmap="gray"), axes[1, 1].set_title("D0 = 160, n = 2")
axes[1, 2].imshow(image6, cmap="gray"), axes[1, 2].set_title("D0 = 460, n = 2")

plot.show()
