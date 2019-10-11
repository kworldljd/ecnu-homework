##2019年10月8日 对一副图像加噪，进行几何均值，算术均值，谐波，逆谐波处理
#1）给图像增加10%的椒盐噪声；添加方差为0.4的高斯噪声的叠加图片。（此部分同第三章，为做区别，本次做高斯与椒盐的叠加噪声）
#2）使用几何平均，f(x,y) = （prod(g(s,t))^(1/mn)
#3）使用编制逆谐波函数，当Q = 0 退化为算术均值，当Q=-1时，退化为谐波均值滤波器
#  (s,t) ∈ Sxy  f(x,y)= sum g(s,t)^(Q+1)  / sum g(s,t)^Q
#4）追加自适应中值滤波效果测试。

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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

saltImg = addSalt(gaussImg, 0.1)

##滤波卷积操作：
# 定义函数指定一个原图，使用规定的3*3模板进行滤过操作。
# 采用复制边方式进行处理。
# method三种方法：1） sum:求和平均数，2）prod：求算术平均数，3）median：求中值。4)antiharm: 逆谐滤波器
def fn_filter33 (orgImg, filterImg, method="sum", Q = 0):
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
    out_img = np.zeros(orgImg.shape)

    #print(orgImg.shape, "开始卷积滤波", "method = %s"%method)

    if method == "sum" :
        for i in np.arange(1, rows+1):
            for j in np.arange(1, cols+1):
                rect = filterImg*extentImg[i-1:i+2, j-1:j+2]   #i+2的原因是切片时截至数不包括在里面
                out_img[i-1, j-1] = np.sum(rect)  #np.uint8(abs(np.sum(rect))) #均值： 如果是十字星型模板，则不能使用mean方法，需要计算sum，除以十字星模板的sum

    elif method == "prod": #几何平均数
        mn = np.count_nonzero(filterImg)
        print("prod method, mn is :", mn)
        for i in np.arange(1, rows + 1):
            for j in np.arange(1, cols + 1):
                rect = filterImg * extentImg[i - 1:i + 2, j - 1:j + 2]  # i+2的原因是切片时截至数不包括在里面

                #tempRect = rect ** (1/mn)  ##先开N次方，否则cumprod可能回超出数值范围。
                ##剔除0点，否则图片会被黑色污染。
                tempLine = np.ravel(rect)
                tempLine =  tempLine[tempLine > 0] #剔除0值
                prod = np.power(tempLine, 1/(tempLine.size) ).cumprod()[-1] #np.cumprod(tempRect)[-1]
                #print(i, j, prod)
                out_img[i - 1, j - 1] = np.uint8( prod )  # 几何平均数
    elif method == "median":
        for i in np.arange(1, rows + 1):
            for j in np.arange(1, cols + 1):
                rect = filterImg * extentImg[i - 1:i + 2, j - 1:j + 2]  # i+2的原因是切片时截至数不包括在里面
                out_img[i - 1, j - 1] = np.uint8(np.median(rect))  # 中值
    elif method == "antiharm": #逆谐滤波器
        if Q >=0 :
            for i in np.arange(1, rows + 1):
                for j in np.arange(1, cols + 1):
                    rect = filterImg * extentImg[i - 1:i + 2, j - 1:j + 2]  # i+2的原因是切片时截至数不包括在里面
                    val = np.power(rect,Q+1).sum()/np.power(rect,Q).sum()
                    out_img[i - 1, j - 1] = np.uint8(val)  #
        elif Q <= -1:
            pind = Q*-1
            for i in np.arange(1, rows + 1):
                for j in np.arange(1, cols + 1):
                    rect = filterImg * extentImg[i - 1:i + 2, j - 1:j + 2]  # i+2的原因是切片时截至数不包括在里面
                    val = np.power(rect,pind).sum()/np.power(rect,pind-1).sum()
                    out_img[i - 1, j - 1] = np.uint8(val)  # 中值
    return out_img


####2. 平滑滤波处理。
#3*3的均值滤波器
templateMean = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])

#中值滤波
print("start sharpen image by median algorithm")
medianImg = fn_filter33(saltImg, templateMean, method="median")
print("start sharpen image by mean algorithm")
meanImg = fn_filter33(saltImg, templateMean, method="antiharm", Q=0)
prodMeanImg = fn_filter33(saltImg,templateMean, method="prod")
print(prodMeanImg)
#谐滤波
harmImg = fn_filter33(saltImg, templateMean, method="antiharm",Q=-1)
#逆谐滤波 Q为正，清除胡椒噪声。
antiharmImg = fn_filter33(saltImg,templateMean,method="antiharm", Q=2)

f, axes = plt.subplots(2, 3)
axes[0, 0].set_title("原图")
axes[0, 0].imshow(img, cmap="gray")
axes[0, 1].set_title("高斯及椒盐噪声后图像")
axes[0, 1].imshow(saltImg, cmap="gray")
axes[0, 2].set_title("几何均值滤波后图像")
axes[0, 2].imshow(prodMeanImg, cmap="gray")

axes[1, 0].set_title("均值滤波器（逆谐滤波Q=0）后图像")
axes[1, 0].imshow(meanImg, cmap="gray")
axes[1, 1].set_title("谐滤波图像")
axes[1, 1].imshow(harmImg, cmap="gray")
axes[1, 2].set_title("逆谐波过滤器 Q=2 图像")
axes[1, 2].imshow(antiharmImg, cmap="gray")

plt.show()

