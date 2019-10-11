###第四周作业：对一副图像进行傅立叶变换，显示频谱，取其5，50，150为截至频率，进行频率域平滑，锐化，显示图像。
# #1）傅里叶变换：直接调用numpy的fft的2维傅里叶变换fft2,并作居中处理
# #2）显示频谱：取幅度谱，求对数，并通过线性变换至返回【0，255】，显示该图像
#    （求对数的原因是幅度谱的值比较大，超出显示范围，线性变换的原因是取完对数，范围比较小【0，20】，图像显示不易观察
# 3）取D0={5，50，150}，设置滤波器H（理想，butterworth ，gauss），显示图像。步骤：
#   HUV = 1/（1+(DUV/D0)^2n)   butterworth 低通滤波器公式，
#   HUV = exp(-DUV^2/2D0^2)    guass 低通滤波公式
#   高通 HP = 1- HL
#   滤波操作 GUV = HUV*FUV
#   傅里叶反变换（先将中心点切换回左端点），取出实部
#   对实部进行范围变换，线性到【0，255】范围，显示图片。
# 参考：https://blog.csdn.net/cjsh_123456/article/details/79342300

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#线性变换到【0-255】之间
def line2Scale(originalImg):
    scale = 0
    convertedImg = np.zeros(originalImg.shape)
    if(originalImg.min()<0):
        scale = originalImg.max() - originalImg.min()
        convertedImg = originalImg + np.array([-1*originalImg.min()]* originalImg.size).reshape(originalImg.shape) #使最小值为0
    else:
        scale = originalImg.max()
        convertedImg = originalImg

    convertedImg = np.uint8(convertedImg/scale * 255)
    return convertedImg


#butterworth 低通滤波器：中心在(m + 1, n + 1)
def func_LowerFilter(FUV, D0, ng, model="butterworth", isLower = True):
    HFilter = np.zeros(FUV.shape)
    m = FUV.shape[0] // 2
    n = FUV.shape[0] // 2
    print("m, n ", m, n)
    if isLower == True :
        if model == "butterworth"  :
            for u in np.arange(0, FUV.shape[0]):
                for v in np.arange(0, FUV.shape[1]):
                    duv = (u-m)**2 +(v-n)**2
                    HFilter[u, v] = 1/(1+(duv**ng)/(D0**(2*ng)))
    else:
        if model == "butterworth"  :
            for u in np.arange(0, FUV.shape[0]):
                for v in np.arange(0, FUV.shape[1]):
                    duv = (u-m)**2 +(v-n)**2
                    HFilter[u, v] =1- 1/(1+(duv**ng)/(D0**(2*ng)))
    GUV = HFilter * FUV

    #反傅里叶变换,先将中心返回到左上角
    gxy = np.fft.ifft2(np.fft.ifftshift(GUV))
    gReal = np.real(gxy)
    print("greal: ", gReal.min(), gReal.max())
    #获取图片：做归一化，转成【0，255】

    img_out = line2Scale(gReal)
    #print("img_out: ", img_out.min(), img_out.max())
    return img_out


#读取图片，并进行傅里叶变换，显示频谱。
img = cv.imread("./source/lena.jpg",cv.IMREAD_GRAYSCALE)
#
# m, n = img.shape[0], img.shape[1]
# centerImg = np.zeros((m*2, n*2))
# for i in np.arange(0, m):
#     for j in np.arange(0, n):
#         centerImg[i, j] = img[i, j]*((-1)**(i+j))

#快速傅里叶变换算法,并进行居中处理
FFTImg = np.fft.fft2(img)
fshift = np.fft.fftshift(FFTImg)
#展示频谱fft结果是复数, 求幅度谱，并对其求对数
fimg = np.log(np.abs(fshift))

print("fimg的最小、最大值: ", fimg.min(),fimg.max(),fimg.size,fimg.shape)

fimgFix = line2Scale(fimg)
# #fimgFix = np.uint8(fimg)
fimgFix2 = np.uint8(np.ones_like(fimg) * 255 - fimg)
#
# cv.imshow("fix1", fimgFix)
# cv.imshow("fix2", fimgFix2)

figure, axes = plt.subplots(2, 2)
axes[0, 0].set_title("orgImg")
axes[0, 0].imshow(img, cmap="gray")
axes[0, 1].set_title("Fourier frequency image by linal exchange to [0,255]")
axes[0, 1].imshow(fimgFix, cmap="gray")

axes[1, 0].set_title("Fourier frequency original image hist")
axes[1, 0].hist(fimgFix)

axes[1, 1].set_title("Fourier frequency original image convert")
axes[1, 1].imshow(fimgFix2, cmap="gray")

fliterFigure, fliterAxes = plt.subplots(2, 3)
ipos = 0
print("star lower filter ")
#5,50,150 低通巴特沃斯滤波
for D0 in np.array([5,50,150]):
  lowerImg = func_LowerFilter(fshift, D0, 2)
  fliterAxes[0, ipos].set_title("lower filter by %d" %D0)
  fliterAxes[0, ipos].imshow(lowerImg, cmap="gray")
  ipos += 1

ipos = 0
print("start higher filter ")
#5,50,150 高通巴特沃斯滤波
for D0 in np.array([5,50,150]):
  lowerImg = func_LowerFilter(fshift, D0, 2, isLower=False)
  fliterAxes[1, ipos].set_title("Higher filter by %d" %D0)
  fliterAxes[1, ipos].imshow(lowerImg, cmap="gray")
  ipos += 1

fliterFigure.suptitle("filter by butterworth algorithm")

plt.show()
