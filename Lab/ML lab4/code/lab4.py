import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 实验配置
num = 200  # 生成的原样本点数
dim = 3  # 原样本点维度
new_dim = 1  # PCA后样本点维度

method = 2  # 1:自己生成数据进行PCA 2:利用人脸数据进行PCA

face_dim = 10  # 人脸图像要降至的维度
filepath = ".\\Face"  # 人脸数据图片的路径
size = (60, 60)  # 设置统一图片大小


def gen_data(num, dim):
    "生成数据"
    data = np.zeros((dim, num))

    if dim == 2:
        mean = [2, -2]
        cov = [[1, 0], [0, 0.01]]
    elif dim == 3:
        mean = [2, 2, 3]
        cov = [[1, 0, 0], [0, 1, 0], [0, 0, 0.01]]

    data = np.random.multivariate_normal(mean, cov, num)
    # print(data.shape) # (100,2)
    return data


def PCA(data, new_dim):
    "实现PCA算法"
    x_mean = np.sum(data, axis=0) / num  # 求均值
    decentral_data = data - x_mean  # 中心化
    cov = decentral_data.T @ decentral_data  # 计算协方差
    eigenvalues, eigenvectors = np.linalg.eig(cov)  # 特征值分解
    eigenvectors = np.real(eigenvectors)
    dim_order = np.argsort(eigenvalues)  # 按从小到大获得特征值的索引
    PCA_vector = eigenvectors[:, dim_order[:-(new_dim + 1):-1]] # 选取最大的特征值对应的特征向量
    x_pca = decentral_data @ PCA_vector @ PCA_vector.T + x_mean  # 计算PCA之后的x值
    return PCA_vector, x_mean, x_pca


def PCA_show():
    "可视化PCA结果"
    data = gen_data(num, dim)  # 生成数据
    PCA_vector, x_mean, x_pca = PCA(data, new_dim)  # 执行PCA算法

    # 绘制散点图
    if dim == 2:  # 维数为2
        plt.scatter(data.T[0], data.T[1], c='w',
                    edgecolors='#86A697', s=20, marker='o', label='origin data')
        plt.scatter(x_pca.T[0], x_pca.T[1], c='w',
                    edgecolors='#D75455', s=20, marker='o', label='PCA data')

    elif dim == 3:  # 维数为3
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(data.T[0], data.T[1], data.T[2],
                   c='#86A697', s=20, label='origin data')
        ax.scatter(x_pca.T[0], x_pca.T[1], x_pca.T[2],
                   c='#D75455', s=20, label='PCA data')

    plt.title("num = %d, dim = %d, new dim = %d" % (num, dim, new_dim))
    plt.legend()
    plt.show()
    return


def read_data():
    "读入人脸数据并展示"
    img_list = os.listdir(filepath)  # 获取文件名列表
    data = []
    i = 1
    for img in img_list:
        path = os.path.join(filepath, img)
        plt.subplot(3, 3, i)
        with open(path) as f:
            img_data = cv2.imread(path)  # 读取图像
            img_data = cv2.resize(img_data, size)  # 压缩图像至size大小
            img_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)  # 三通道转换为灰度图
            plt.imshow(img_gray)  # 预览
            h, w = img_gray.shape
            img_col = img_gray.reshape(h * w)  # 对(h,w)的图像数据拉平
            data.append(img_col)
        i += 1
    plt.show()

    return np.array(data)  # (9, 1600)


def SNR(img1, img2):
    "计算信噪比"
    diff = (img1 - img2) ** 2
    mse = np.sqrt(np.mean(diff))
    return 20 * np.log10(255.0 / mse)


def face_show():
    "人脸数据PCA"
    data = read_data()
    n, pixel = data.shape
    PCA_vector, x_mean, x_pca = PCA(data, face_dim)
    x_pca = np.real(x_pca)  # 仅保留实部
    # 绘制PCA后的图像
    plt.figure()
    for i in range(n):
        plt.subplot(3, 3, i+1)
        plt.imshow(x_pca[i].reshape(size))
    plt.show()
    # 计算信噪比
    print("压缩后维度为 %d，信噪比如下：" % face_dim)
    for i in range(n):
        snr = SNR(data[i], x_pca[i])
        print("图 %d, 信噪比: %.3f" % (i+1, snr))
    return


if method == 1:
    PCA_show()
elif method == 2:
    face_show()
