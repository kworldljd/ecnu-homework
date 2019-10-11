###2019年10月7日 ：提取一副图像中的红色，用HIS模型处理。
#1）使用opencv的cvtColor方法将RGB图像转换成HSL图像，并提取其中的H，S分量，并从H中获取红色（范围在【0，10】，【156，180】设置为1，
#      其他为0，色相范围参考：https://zh.wikipedia.org/wiki/%E8%89%B2%E7%9B%B8#%E8%89%B2%E7%9B%B8%E7%8E%AF, https://www.cnblogs.com/wangyblzu/p/5710715.html）
#   注：H的正常范围是【0，360】，opencv为了适配到【0，255】范围，H的范围减半。详见：https://docs.opencv.org/master/de/d25/imgproc_color_conversions.html#color_convert_rgb_hls
#2) 使用饱和度分量制作掩模（二值化，阈值取最大饱和度的50%，饱和度低于50%的点舍去）template
#3) H分量与template进行乘积，得到提取的图片。

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot

img_BGR = cv.imread("./source/redApple.jpg")  #读取格式默认是BGR格式
img_HLS = cv.cvtColor(img_BGR, cv.COLOR_BGR2HLS)
HvalueTemp,  Lvalue, Svalue = cv.split(img_HLS) #获得三个分量
#从H分量中提取红色
cond = np.logical_or(np.logical_and(HvalueTemp>= 0, HvalueTemp<= 10), np.logical_and(HvalueTemp>= 156, HvalueTemp<= 180))
Hvalue = np.where(cond, 255, 0)
figure, axes = plot.subplots(2, 3)
axes[0, 0].set_title("original image")
axes[0, 0].imshow(cv.cvtColor(img_BGR, cv.COLOR_BGR2RGB))
axes[0, 1].set_title("red selected in Hvalue[0,10],[156,180] ")
axes[0, 1].imshow(np.uint8(Hvalue), cmap="gray")
axes[0, 2].set_title("Svalue image")
axes[0, 2].imshow(Svalue, cmap="gray")

axes[1, 0].set_title("Lvalue image")
axes[1, 0].imshow(Lvalue, cmap="gray")

# cv.imshow("Hvalue", np.uint8(Hvalue))
# cv.imshow("Svalue", Svalue)
# cv.imshow("Lvalue", Lvalue)

print("Hvalue", Hvalue)
print("Svalue", Svalue)
print("Lvalue", Lvalue)


threshold = Svalue.max()*0.5 #舍去饱和度低于最大饱和度50%的点。
print("threshold", threshold)
template = np.where(Svalue > threshold, 1, 0)
print("template", template)
product =np.uint8(Hvalue*template)
# cv.imshow("template", np.uint8(template*255))
# cv.imshow("product", product)
# print("product", product)
axes[1, 1].set_title("template made by Svalue threshold=50%")
axes[1, 1].imshow(template, cmap="gray")

axes[1, 2].set_title("product image by Hvalue X template")
axes[1, 2].imshow(product, cmap="gray")


#plot.hist(product)
plot.show()

cv.waitKey(0)
