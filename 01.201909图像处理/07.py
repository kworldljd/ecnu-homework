#7.	对一副二值图像进行膨胀、腐蚀、开、闭操作
#生成5X5的核，对图像进行膨胀操作，外围变大，与操作，全为0的为0，否则为1
#生成3X3的核，对图像进行腐蚀操作，剔除毛刺，与操作，存在0的为0，否则为1
#开操作：先腐蚀后膨胀 (A-B)+B 剔除毛边后复原
#闭操作：先膨胀后腐蚀 (A+B)-B

#参考：https://blog.csdn.net/Eastmount/article/details/83581277

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#膨胀操作
img = cv.imread("./source/dilate0.jpg", cv.IMREAD_GRAYSCALE)

dilateKernal = np.ones((5,5), dtype=np.uint8)
dilateImg1 = cv.dilate(img, dilateKernal)
dilateImg5 = cv.dilate(img, dilateKernal, iterations=5)

#腐蚀操作
img2 = cv.imread("./source/erode0.jpg", cv.IMREAD_GRAYSCALE)
erodeKernal = np.ones((3,3), dtype=np.uint8)
erodeImg1 = cv.erode(img2, erodeKernal)
erodeImg2 = cv.erode(img2, erodeKernal, iterations=5)

figure, axes = plt.subplots(3,3)
for ax in axes.flat:  #隐藏x，y轴
    ax.set(xticks=[], yticks=[])

axes[0, 0].imshow(img, cmap="gray") , axes[0, 0].set_title("膨胀原图")
axes[0, 1].imshow(dilateImg1, cmap="gray"), axes[0, 1].set_title("膨胀迭代1")
axes[0, 2].imshow(dilateImg5, cmap="gray"), axes[0, 2].set_title("膨胀迭代5")
axes[1, 0].imshow(img2, cmap="gray"), axes[1, 0].set_title("腐蚀原图")
axes[1, 1].imshow(erodeImg1, cmap="gray"), axes[1, 1].set_title("腐蚀迭代1")
axes[1, 2].imshow(erodeImg2, cmap="gray"), axes[1, 2].set_title("腐蚀迭代5")

##开闭操作。
#开操作：
fingerImg = cv.imread("./source/fingerprint.jpg", cv.IMREAD_GRAYSCALE)
openErodeFingerImg = cv.erode(fingerImg,erodeKernal, iterations=2)
openFingerImg = cv.dilate(openErodeFingerImg, erodeKernal, iterations=2)
axes[2, 0].imshow(fingerImg, cmap="gray"),      axes[2, 0].set_title("开操作原图")
axes[2, 1].imshow(openFingerImg, cmap="gray"),  axes[2, 1].set_title("开操作（腐蚀，膨胀2次）")
#对开操作图像进行闭操作。
axes[2, 2].imshow( cv.erode( cv.dilate(openFingerImg, erodeKernal, iterations=1), erodeKernal), cmap="gray"), axes[2,2].set_title("闭操作（膨胀、腐蚀1次)")
# cv.imshow("erode", openFingerImg)
# cv.imshow("dilate", cv.dilate(openFingerImg, erodeKernal, iterations=2))

#plt.axes('off')
plt.show()