###2019年9月14日 打开一幅低对比度图像，拉伸其图像，直方图均衡。
#使用skimage开发包 和 根据原理自行开发，并对比显示其效果。
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure


img = cv.imread("./source/02.lowcontrast.jpg",cv.IMREAD_GRAYSCALE)  #读入做灰色处理
print(img.shape)



#使用skimage的伽马变换
gamma_img = exposure.adjust_gamma(img, 5)
#根据原理开发的伽马变换：对图像拉伸，先需做归一化(/255去除量纲），然后进行变换，最后复原回图像空间（需做无符号转换）
#曝光过度，调黑。尝试γ取2，3，4，5，检查效果
stretch2 = np.uint8(np.power(img / 255.0, 2) * 255.0)
stretch3 = np.uint8(np.power(img/255.0,3)*255.0)
stretch4 = np.uint8(np.power(img/255.0,4)*255.0)
stretch5 = np.uint8(np.power(img/255.0,5)*255.0)
print(stretch2)
#对比skimage与自行开发的伽马变换。
#cv.imshow("dev and skimage",np.hstack((stretch15,gamma_img)))
figure,axs =plt.subplots(2,3)
axs[0,0].set_title("original image")
axs[0,0].imshow(img,cmap='gray') #原图

axs[0,1].set_title("stretch 2")
axs[0,1].imshow(stretch2, cmap='gray')

axs[0,2].set_title("stretch 3")
axs[0,2].imshow(stretch3,cmap='gray')

axs[1,0].set_title("stretch 4")
axs[1,0].imshow(stretch4,cmap='gray')

axs[1,1].set_title("stretch 5")
axs[1,1].imshow(stretch5,cmap='gray')

axs[1,2].set_title("package skimage stretch 5")
axs[1,2].imshow(gamma_img,cmap='gray')


#的直方图均衡函数
equ = cv.equalizeHist(img)
#cv.imshow('equalization by opencv', np.hstack((img, equ)))  # 并排显示

#按照公式计算：
#原始图像灰度级统计
rk = np.bincount(img.ravel(), minlength=256)  #图像二维数组先转换为一维，然后使用bincount求出不同像素级数的个数
prk=rk/img.size  #各级像素的概率
sk=np.zeros(256)
pos = 0
sum = 0.0
for val in prk:
    sum = sum + val  #累计
    sk[pos] = sum
    pos = pos+1

#展示处理后的直方图
#plt.figure()
#plt.plot(sk)

print(sk,pos,sum)

#sk取整扩展
skInt = np.uint8(255*sk+0.5)

print(skInt)

def fn_change(x):
    return skInt[x]

equImg2=np.array(list(map(fn_change,img)),dtype=np.uint8)
equFigure,equAxs =plt.subplots(2,3)
equAxs[0,0].set_title("original image")
equAxs[0,0].imshow(img,cmap='gray') #原图

equAxs[0,1].set_title("equalization")
equAxs[0,1].imshow(equImg2, cmap='gray')

equAxs[0,2].set_title("equalization by opencv")
equAxs[0,2].imshow(equ, cmap='gray')

#展示图像直方图
# plt.hist(img.ravel(), 256, [0, 256])#等价于 bincount先求出每个级数的个数，然后画直方图。
hist =  np.bincount(img.ravel(), minlength=256)/img.size
equAxs[1,0].set_title("orignal image hist")
equAxs[1,0].plot(hist) #原图

hist2 =  np.bincount(equImg2.ravel(), minlength=256)/img.size
equAxs[1,1].set_title("equ image hist")
equAxs[1,1].plot(hist2)

hist3 =  np.bincount(equ.ravel(), minlength=256)/img.size
equAxs[1,2].set_title("equ by openev hist")
equAxs[1,2].plot(hist3)

plt.show()

cv.waitKey(0)


