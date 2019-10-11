#2019年9月17日 对一副图像加噪声，进行平滑，锐化作用，处理步骤：
#1.高斯噪声：生成正态分布的随机数，期望为0，方差=0.04（方差太大，图像噪声太多，太小，噪声不明显），矩阵大小与图像大小相同；随机矩阵+原图（归一化）后范围限制在0~1之间，然后进行还原为0~255的灰度值。
#2.添加椒盐噪声：产生x%的椒盐噪声：设总的像素点m，则x%的椒盐量，总共需要点数量为n= m*x; 随机生成n个点坐标，随机设置该处的值为0或255.
#3.平滑处理：高斯噪声的平滑使用，去除椒盐噪声。
#4.锐化处理：梯度锐化，roberts，sobel，Laplace算子锐化
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure


img = cv.imread("./source/lena.jpg",cv.IMREAD_GRAYSCALE)

#高斯噪声
noise=np.random.normal(0,0.04,size=img.shape) #产生高斯（正态）分布噪声，期望为0，方差=0.04 （方差过大1，图像噪声太多，方差过小0，噪声太少）
gaussImg = np.uint8(np.clip((noise + img / 255), 0, 1) * 255)  #范围在0，1之间，之外的
#cv.imshow("gauss",np.hstack((img,gaiImg)))

##2019年9月28日 优化椒盐噪声的生成
# 产生x%的椒盐噪声：设总的像素点m，则x%的椒盐量，总共需要点数量为n= m*x; 随机生成n个点坐标，随机设置该处的值为0或255.
def addSalt(orgImg, x):
    m = orgImg.size
    n = np.int(m*x)
    pos = zip(np.random.randint(0, orgImg.shape[0], size=n), np.random.randint(0, orgImg.shape[1], size= n))
    newImg = np.copy(orgImg)
    for x, y in pos:
        if(np.random.randint(2)):
            newImg[x, y] = 0
        else:
            newImg[x, y] = 255
    return newImg

#椒盐噪声,随机产生0~255之间的数值，然后进行二值化，最后与原图合并
# 2019年9月28日 注释：无法确定比例，注释，改为新的函数，增加椒盐值比例。
# maxthreshold = 200 #阈值100，随机数小于100的，设置为黑色
# minthreshold = 50
# salt = np.random.randint(0,255,img.shape,dtype=np.uint8) #生成随机数
# saltImg = salt.ravel()
# imgSeq = img.ravel()
# pos = 0
#
# for val in saltImg:
#     if val > maxthreshold :  ##阈值大于阀直的不变，小于阈值的置黑0
#         saltImg[pos]=255
#     elif val < minthreshold:
#         saltImg[pos] = 0
#     else:
#         saltImg[pos] = imgSeq[pos]
#     pos += 1
# saltImg.resize(img.shape)
saltImg = addSalt(img, 0.1)

#cv.imshow("salt",np.hstack((img,saltImg)))


##滤波卷积操作：
# 定义函数指定一个原图，使用规定的3*3模板进行滤过操作。
# 采用复制边方式进行处理。
def fn_filter33 (orgImg, filterImg, method="sum"):
    rows,cols = orgImg.shape
    #为原图扩展边，以便于计算，扩展方式为边复制.
    extentImg = np.zeros((rows+2, cols+2), dtype=np.uint8)
    extentImg[1:rows+1, 1:cols+1] = orgImg #中间部分用原图填充
    extentImg[0, 1:cols+1]=orgImg[0, :] #第一行复制img的第一行
    extentImg[rows+1, 1:cols+1]=orgImg[rows-1, :]#最后一行复制img的最后一行
    extentImg[1:rows+1, 0]=orgImg[:, 0]  #第一列复制过来
    extentImg[1:rows+1, cols+1]=orgImg[:, cols-1] #最后一列
    #补齐四个角
    extentImg[0, 0]=orgImg[0, 0]
    extentImg[0, cols+1] = orgImg[0, cols-1]
    extentImg[rows+1, 0] = orgImg[rows-1, 0]
    extentImg[rows+1, cols+1] = orgImg[rows-1, cols-1]

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

    # f, axes = plt.subplots(1, 3)
    # axes[0].set_title("orgImg")
    # axes[0] = plt.imshow(orgImg, cmap="gray")
    # axes[1].set_title("mean")
    # axes[1] = plt.imshow(meanImg, cmap="gray")
    # axes[2].set_title("median")
    # axes[2] = plt.imshow(medianImg, cmap="gray")
    # plt.show()
    # cv.imshow("org", orgImg)
    # cv.imshow("mean", meanImg)
    # cv.imshow("median", medianImg)


####2. 平滑滤波处理。
#3*3的均值滤波器
templateMean = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])

#中值滤波
print("start sharpen image by median algorithm")
medianImg = fn_filter33(saltImg, templateMean, method="median")
#均值滤波
print("start sharpen image by mean algorithm")
meanImg = fn_filter33(saltImg, templateMean*1/9, method="sum")


f, axes = plt.subplots(2, 3)
axes[0, 0].set_title("orgImg")
axes[0, 0].imshow(img, cmap="gray")
axes[0, 1].set_title("gaussian  noise")
axes[0, 1].imshow(gaussImg, cmap="gray")
axes[0, 2].set_title("spiced salt 10%")
axes[0, 2].imshow(saltImg, cmap="gray")

axes[1, 0].set_title("mean filter3*3 on salt image")
axes[1, 0].imshow(meanImg, cmap="gray")
axes[1, 1].set_title("median filter3*3  on salt image ")
axes[1, 1].imshow(medianImg, cmap="gray")

###3. 锐化处理：包括：1阶的：梯度法，Roberts算法（交叉差分），sobel算法，prewitt算子（对噪声敏感）：突出小缺陷，去除慢变化背景
# 2阶的拉普拉斯算子：增强灰度突变处的比对度。

#3.1 梯度法，roberts
#梯度法为：newPixel = abs(fn[i, j+1] - fn[i, j]) + abs( fn[i+1, j] - fn[i, j])
#         [-1, 1],
#         [1,   0]
#roberts: newPixel = abs(fn[i+1,j+1] - fn[i,j]) + abs(fn[i+1,j] - fn[i, j+1])
#        [-1, -1]
#        [1,   1]
print(" start sharpen image by gradient and roberts algorithm")
gradientImg = np.zeros(img.shape, dtype=np.uint8)
robertsImg = np.zeros(img.shape, dtype=np.uint8)
for i in np.arange(0,img.shape[0]-1):
    for j in np.arange(0, img.shape[1]-1):
        grandient = abs(int(img[i, j+1]) - int(img[i, j])) + abs(int(img[i+1, j]) - int(img[i, j])) #超过255怎么处理？  #要先转换类型为有符号，否则计算有偏差，按无符号处理。
        gradientImg[i, j] = 255 if grandient > 255 else grandient

        roberts = abs(int(img[i+1, j+1]) - int(img[i, j])) + abs(int(img[i+1, j]) - int(img[i, j+1]))
        robertsImg[i, j] = 255 if roberts > 255 else roberts

#图像取反
def fn_convert(orgImg):
    convImg = np.array([255]*orgImg.size).reshape(orgImg.shape)
    convImg = convImg - orgImg
    return convImg

###sobel算法：检测边缘较平滑，光洁
## newPixel = max(HorizontalDifference , VerticalDifference)
## HorizontalDifference = abs( fn[
# 结构：horizontalDifference：
#       [-1, 0, 1]
#       [-2, 0, 2]
#       [-1, 0, 1]
#  verticalDifference： 将HorizontalDifference 顺时针旋转90度

#Prewitt算子：
#结构：
#       [-1, 0, 1]
#       [-1, 0, 1]
#       [-1, 0, 1]

print(" start sharpen image by prewitt and sobel algorithm")
templateSobelHor = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])
templateSobelVer = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]])
print("compute sobel horizontalDifference ")
sobelHorImg = abs(fn_filter33(img, templateSobelHor))
sobelHorImg = np.uint8((sobelHorImg/np.max(sobelHorImg))* 255) #归一化处理，范围到【0，255】之间
print("compute sobel verticalDifference ")
sobelVerImg = abs(fn_filter33(img, templateSobelVer))
sobelVerImg = np.uint8((sobelVerImg/np.max(sobelVerImg))* 255) #归一化处理，范围到【0，255】之间

#取两者中的较大值作为新的值。
sobelImg = np.zeros(sobelHorImg.shape, dtype=np.uint8)
for i in np.arange(0, sobelHorImg.shape[0]):
    for j in np.arange(0, sobelHorImg.shape[1]):
        sobelImg[i, j] = sobelHorImg[i, j] if sobelHorImg[i, j] > sobelVerImg[i, j] else sobelVerImg[i, j]

###拉普拉斯算子：
#拉普拉斯边缘检测
print("start sharpen image by  laplace algorithm")
templateLaplace = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])
templaplaceImg = fn_filter33(img, templateLaplace)
#处理后的图+254，并限制范围在0，255之间，效果并不好。改为将templaplaceImg的值范围重新度量，归一化处理，压缩至【0，255】之间。效果偏暗。
print("templaplaceImg min, max = ",templaplaceImg.min(), templaplaceImg.max())
#方法一：灰度级偏移254.
laplaceImgshift = np.clip(templaplaceImg+ np.array([254]*templaplaceImg.size).reshape(templaplaceImg.shape), 0, 255).astype(np.uint8)
#方法二：归一化到【0，255】之间
scale = 0 #templaplaceImg的值域范围。
if(templaplaceImg.min()<0):
    scale = templaplaceImg.max() - templaplaceImg.min()
    templaplaceImg = templaplaceImg + np.array([-1*templaplaceImg.min()]* templaplaceImg.size).reshape(templaplaceImg.shape) #使最小值为0
else:
    scale = templaplaceImg.max()

if(scale>255):
    laplaceImg = np.uint8(templaplaceImg/scale * 255)

#对比偏移处理方法的效果。

f2, axesSharpen = plt.subplots(2, 3)
axesSharpen[0, 0].set_title("original Image")
axesSharpen[0, 0].imshow(img , cmap="gray")
axesSharpen[0, 1].set_title("gradient Image")
axesSharpen[0, 1].imshow(fn_convert(gradientImg) , cmap="gray")
axesSharpen[0, 2].set_title("roberts Image")
axesSharpen[0, 2].imshow(fn_convert(robertsImg), cmap="gray")

axesSharpen[1, 0].set_title("sobel Image")
axesSharpen[1, 0].imshow(fn_convert(sobelImg), cmap="gray")

axesSharpen[1, 1].set_title("laplacian Image(by uniformization)")
axesSharpen[1, 1].imshow(laplaceImg, cmap="gray")


axesSharpen[1, 2].set_title("laplacian Image(by add 254 grayscale)")
axesSharpen[1, 2].imshow(laplaceImgshift, cmap="gray")

plt.show()


####试验2019年9月30日  laplacian算子图像增强。（也即 unsharp masking）

cv.waitKey(0)


