#2019年10月3日
#第三章ppt，p135 计算。
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

##滤波卷积操作：
# 定义函数指定一个原图，使用规定的3*3模板进行滤过操作。
# 采用复制边方式进行处理。
def fn_filter33 (orgImg, filterImg, method="sum", edgeModel="copy"):
    rows,cols = orgImg.shape
    #为原图扩展边，以便于计算，扩展方式为边复制.
    extentImg = np.zeros((rows+2, cols+2), dtype=np.uint8)
    extentImg[1:rows+1, 1:cols+1] = orgImg #中间部分用原图填充

    if(edgeModel == "copy"):
        extentImg[0, 1:cols+1]=orgImg[0, :] #第一行复制img的第一行
        extentImg[rows+1, 1:cols+1]=orgImg[rows-1, :]#最后一行复制img的最后一行
        extentImg[1:rows+1, 0]=orgImg[:, 0]  #第一列复制过来
        extentImg[1:rows+1, cols+1]=orgImg[:, cols-1] #最后一列
        #补齐四个角
        extentImg[0, 0]=orgImg[0, 0]
        extentImg[0, cols+1] = orgImg[0, cols-1]
        extentImg[rows+1, 0] = orgImg[rows-1, 0]
        extentImg[rows+1, cols+1] = orgImg[rows-1, cols-1]
    else:#添加0边
        extentImg[0,:] = np.array([0]*(cols+2))
        extentImg[rows+1, :] = extentImg[0,:]
        extentImg[:, 0] = np.array([0]*(rows+2)).reshape((rows+2, 1))[:,0]
        extentImg[:, cols+1] =extentImg[:, 0]

    ##均值滤波图像,中值滤波图像结果
    meanImg = np.zeros(orgImg.shape)
    medianImg = np.zeros(orgImg.shape, dtype=np.uint8)

    #print(orgImg.shape, "开始卷积滤波", "method = %s"%method)

    if method == "sum" :
        for i in np.arange(1, rows+1):
            for j in np.arange(1, cols+1):
                rect = filterImg*extentImg[i-1:i+2, j-1:j+2]   #i+2的原因是切片时截至数不包括在里面
                meanImg[i-1, j-1] = np.sum(rect)  #np.uint8(abs(np.sum(rect))) #均值： 如果是十字星型模板，则不能使用mean方法，需要计算sum，除以十字星模板的sum
        return meanImg
    else:
        for i in np.arange(1, rows + 1):
            for j in np.arange(1, cols + 1):
                rect = filterImg * extentImg[i - 1:i + 2, j - 1:j + 2]  # i+2的原因是切片时截至数不包括在里面
                medianImg[i - 1, j - 1] = np.uint8(np.median(rect))  # 中值

        return medianImg

#3*3的均值滤波器
templateMean = np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]])
originalImg = np.array([[1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 1]])
#均值滤波
print("copy edge image ")
copyEdgeImg = fn_filter33(originalImg, templateMean, method="sum")
print(copyEdgeImg)
print("add zeros edge image")
addZerosImg = fn_filter33(originalImg, templateMean, edgeModel="zero")
print(addZerosImg)


print("test laplace ")
img = cv.imread("./source/lena.jpg",cv.IMREAD_GRAYSCALE)

####2019年10月5日  测试：拉普拉斯增强：
templateLaplace2 = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])
templaplaceImg2 = fn_filter33(img, templateLaplace2)
print("the max and min of image pixcel: ", templaplaceImg2.max(),templaplaceImg2.min())
plt.hist(templaplaceImg2)
lapaceImg = np.uint8(templaplaceImg2.clip(0, 255))
cv.imshow("laplace img", lapaceImg)
plt.show()


