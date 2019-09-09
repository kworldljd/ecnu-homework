###2019年9月9日 第一周作业：打开图像，显示图像，存储图像，对一张图片进行缩放，观察其分辨率，降低灰度处理
import cv2 as cv
import time
import numpy as np

img = cv.imread("./source/lena.jpg",cv.IMREAD_GRAYSCALE)  #读入做灰色处理
print(img.shape) #打印大小
(height,width) = img.shape

cv.imshow("lena",img)  #显示
rename = str(time.time())
print(rename,height,width)
cv.imwrite("./dest/lena2"+rename+".jpg",img) #存错

#放大一倍
largedim = (int(height*1.5),int(width*1.5))
img_enlarge=cv.resize(img,largedim) #1data 2 (高，宽)
cv.imshow("large",img_enlarge);
print(img_enlarge.shape)

###缩小一半,方法一
img_enMin=cv.resize(img,None,0.5,0.5,cv.INTER_AREA) #1data 2 (高，宽)
cv.imshow("minor1",img_enMin);
print(img_enMin.shape)

###缩小一半,方法二
mindim = (int(height*0.5),int(width*0.5))
img_enMindim=cv.resize(img,mindim,cv.INTER_AREA) #1data 2 (高，宽)
cv.imshow("minor2",img_enMindim);
print(img_enMindim.shape)


##@改变每个像素的灰度值，每个像素降低100,  生成同大小的二维数组，每个值
subval = np.full(img.shape,100,dtype=np.uint8)
subimg = cv.subtract(img,subval)
cv.imshow("subimg",subimg)
cv.waitKey(0)