#2019年9月17日 对一副图像加噪声，进行平滑，锐化作用
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure


img = cv.imread("./source/lena.jpg",cv.IMREAD_GRAYSCALE)

#高斯噪声
noise=np.random.normal(0,0.04,size=img.shape) #产生高斯（正态）分布噪声，期望为0，方差=0.04 （方差他大1，图像噪声太多，方差过小0，噪声太少）
gaiImg = np.uint8(np.clip((noise+img/255),0,1)*255)  #范围在0，1之间，之外的
cv.imshow("lena",np.hstack((img,gaiImg)))

#椒盐噪声,随机产生0~255之间的数值，然后进行二值化，最后与原图合并
threshold = 100 #阈值100，随机数小于100的，设置为黑色
salt = np.random.randint(0,255,img.shape,dtype=np.uint8) #生成随机数
saltImg = salt.ravel()
imgSeq = img.ravel()
pos = 0

for val in saltImg:
    if val > threshold :  ##阈值大于阀直的不变，小于阈值的置黑0
        saltImg[pos]=imgSeq[pos]
    else:
        saltImg[pos] = 0
    pos += 1

saltImg.resize(img.shape)
cv.imshow("salt",np.hstack((img,saltImg)))

####滤波处理。
#3*3的滤波器
filter33=np.array([[0,1,0],
                   [1,1,1],
                   [0,1,0]],dtype=np.uint8)
rows,cols = img.shape
#为原图扩展边，以便于计算
extentImg = np.zeros((rows+2,cols+2),dtype=np.uint8)
extentImg[1:rows,1:cols] = img
extentImg[0,1:cols]=img[0,:] #第一行等于img的第一行
extentImg[rows+1,1:cols]=img[rows-1,:]#最后一行复制img的最后一行
extentImg[1:rows,0]=img[:,0]  #第一列复制过来
extentImg[1:rows,cols+1]=img[:,cols-1] #最后一列
#补齐四个角
extentImg[0,0]=img[0,0]
extentImg[0,cols+1] = img[0,cols-1]
extentImg[rows+1,0]=img[rows-1,0]
extentImg[rows+1,cols+1] = img[rows-1,cols-1]


##均值滤波图像,中值滤波图像结果
meanImg = np.zeros(img.shape,dtype=np.uint8)
midImg = np.zeros(img.shape,dtype=np.uint8)

for i in rows:
    for j in cols:
        meanVal =  
        midVal =
        meanImg[i,j] =


cv.waitKey(0)


# f,axes = plt.subplots(1,3)
# axes[0]=plt.imshow(img,cmap="gray")
# axes[1]=plt.imshow(combImg,cmap="gray")
# cv.imshow("noise",combImg)
# plt.show();