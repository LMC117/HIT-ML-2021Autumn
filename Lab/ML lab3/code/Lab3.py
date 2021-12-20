import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd

kmeans_epsilon = 1e-8  # K-means算法的迭代误差
GMM_epsilon = 1e-5  # GMM算法的迭代误差
GMM_epoch = 40  # GMM算法的迭代次数

con = 0  # 选择的设置
method = 2 # 选择的方法，0:K-means;     1:GMM;      2:UCI;

colors = [
    '#77969A', '#F7D94C', '#86C166', '#51A8DD', '#E98B2A', '#B481BB', '#F596AA', '#D75455'
]  # 颜色库

# 初始设置
config_0 = {
    'k': 4,  # 聚类数
    'n': 300,  # 每类的样本点数
    'dim': 2,  # 样本点维度
    'mu': np.array([
        [-5, 4], [5, 4], [3, -4], [-5, -5]
    ]),  # 均值
    'sigma': np.array([
        [[2, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 1]], [[3, 0], [0, 2]]
    ])  # 方差
}

config_1 = {
    'k': 8,  # 聚类数
    'n': 300,  # 每类的样本点数
    'dim': 2,  # 样本点维度
    'mu': np.array([
        [-4, 3], [4, 2], [1, -4], [-5, -3], [0, 0], [6, -1], [-1, 8], [7, -4]
    ]),  # 均值
    'sigma': np.array([
        [[2, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 1]], [[3, 0], [0, 2]],
        [[2, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 1]], [[3, 0], [0, 2]]
    ])  # 方差
}

configs = [config_0, config_1]


def gen_data(k, n, dim, mu, sigma):
    "生成数据"
    raw_data = np.zeros((k, n, dim))
    for i in range(k):
        raw_data[i] = np.random.multivariate_normal(mu[i], sigma[i], n)
    data = np.zeros((k * n, dim))
    for i in range(k):
        data[i * n:(i + 1) * n] = raw_data[i]
    return data


def kmeans(data, k, N, dim):
    "K-means算法实现"
    category = np.zeros((N, dim+1))
    category[:, 0:dim] = data[:, :]  # 多出的一维用于保存类别标签
    center = np.zeros((k, dim))  # k行dim列，保存各类中心
    for i in range(k):  # 随机以某些样本点作为初始点
        center[i, :] = data[np.random.randint(0, N), :]

    iter_count = 0  # 迭代次数

    # K-means算法核心
    while True:
        iter_count += 1
        distance = np.zeros(k)  # 保存某一次迭代中，点到所有聚类中心的距离

        for i in range(N):
            point = data[i, :]  # 此次选取，用来计算距离的样本点
            for j in range(k):
                t_center = center[j, :]  # 此次选取，用来计算距离的聚类中心
                distance[j] = np.linalg.norm(
                    point - t_center)  # 更新该样本点到聚类中心的距离
            min_index = np.argmin(distance)  # 找到该点对应的最近聚类中心
            category[i, dim] = min_index

        num = np.zeros(k)  # 保存每个类别的样本点数
        count = 0  # 计数有多少个聚类中心更新的距离小于误差值
        new_center_sum = np.zeros((k, dim))  # 临时变量
        new_center = np.zeros((k, dim))  # 保存此次迭代得到的新的聚类中心

        for i in range(N):
            label = int(category[i, dim])
            num[label] += 1  # 统计各类的样本点数
            new_center_sum[label, :] += category[i, :dim]

        for i in range(k):
            if num[i] != 0:
                new_center[i, :] = new_center_sum[i, :] / \
                    num[i]  # 计算本次样本点得出的聚类中心
            new_k_distance = np.linalg.norm(new_center[i, :] - center[i, :])
            if new_k_distance < kmeans_epsilon:  # 计数更新距离小于误差的聚类中心个数
                count += 1

        if count == k:  # 所有聚类中心的更新距离都小于误差值，结束循环
            return category, center, iter_count, num
        else:  # 否则更新聚类中心
            center = new_center


def calc_acc(num, n, N):
    "计算K-means算法的准确率"
    # 思想：将各类点数统一减去n（最开始分类时每一类的数目），求其中元素绝对值之和，除以二，即为分类错误点的数量
    temp = np.abs(num - n)
    wrong = np.sum(temp) / 2
    return 1 - wrong / N


def kmeans_show(k, n, dim, mu, sigma):
    "K-means算法的结果展示"

    data = gen_data(k, n, dim, mu, sigma)
    # print(data.shape)  # (1200,2)
    N = data.shape[0]  # N为样本点总数

    category, center, iter_count, num = kmeans(data, k, N, dim)
    acc = calc_acc(num, n, N)

    for i in range(N):  # 绘制分为各类的样本点
        color_num = int(category[i, dim] % len(colors))
        plt.scatter(category[i, 0], category[i, 1],
                    c=colors[color_num], s=5)

    for i in range(k):
        plt.scatter(center[i, 0], center[i, 1],
                    c='black', marker='x')  # 绘制各个聚类中心

    print("acc:" + str(acc))

    plt.title("K-means results " + "   config:" +
              str(con) + "   iter:" + str(iter_count))
    plt.show()
    return


def GMM_loss(data, k, N, mu, sigma, alpha):
    "计算GMM算法的loss"
    loss = 0
    for i in range(N):
        temp = 0
        for j in range(k):  # 根据公式计算loss
            temp += alpha[j] * \
                multivariate_normal.pdf(data[i], mean=mu[j], cov=sigma[j])
        loss += np.log(temp)
    return loss


def GMM_estep(data, k, N, mu, sigma, alpha):
    "GMM算法的E-step"
    gamma = np.zeros((N, k))  # 初始化隐变量γ
    for i in range(N):
        marginal_prob = 0
        for j in range(k):
            marginal_prob += alpha[j] * multivariate_normal.pdf(data[i],
                                                                mean=mu[j], cov=sigma[j])
        for j in range(k):
            gamma[i, j] = alpha[j] * multivariate_normal.pdf(data[i],
                                                             mean=mu[j], cov=sigma[j]) / marginal_prob
    return gamma


def GMM_mstep(data, gamma, k, N, dim, mu):
    "GMM算法的M-step"
    new_mu = np.zeros((k, dim))
    new_sigma = np.zeros((k, dim, dim))
    new_alpha = np.zeros(k)

    for i in range(k):
        sum_gamma = 0
        for j in range(N):
            sum_gamma += gamma[j, i]

        # 计算新的 μ
        mu_temp = np.zeros(dim)
        for j in range(N):
            mu_temp += gamma[j, i] * data[j]
        new_mu[i] = mu_temp / sum_gamma

        # 计算新的 σ
        sigma_temp = np.zeros(dim)
        for j in range(N):
            dis = data[j] - mu[i]
            sigma_temp += dis**2 * gamma[j, i]

        new_sigma_temp = np.eye(dim)
        new_sigma_temp[0, 0] = sigma_temp[0]
        new_sigma_temp[1, 1] = sigma_temp[1]
        new_sigma[i] = new_sigma_temp / sum_gamma

        # 计算新的 π
        new_alpha[i] = sum_gamma / N

    return new_mu, new_sigma, new_alpha


def GMM(data, k, N, dim, mu, sigma):
    "GMM算法"
    category = np.zeros(N)  # 初始化每个样本点的类标签
    alpha = np.array([1 / k] * k)  # 初始化第k个高斯模型的权重

    new_mu, new_sigma, new_alpha = mu, sigma, alpha

    count = 0  # 记录真实的迭代次数

    print("Iteration:\t\tlog likelihood")

    # GMM算法核心步骤
    for i in range(GMM_epoch):
        count += 1
        old_loss = GMM_loss(data, k, N, new_mu, new_sigma, new_alpha)
        gamma = GMM_estep(data, k, N, new_mu, new_sigma, new_alpha)
        new_mu, new_sigma, new_alpha = GMM_mstep(
            data, gamma, k, N, dim, new_mu)
        new_loss = GMM_loss(data, k, N, new_mu, new_sigma, new_alpha)

        if i % 1 == 0:
            print(str(i) + "\t\t\t" + str(new_loss))
            argmax = np.argmax(gamma, axis=1)
            for j in range(N):
                category[j] = argmax[j]  # 更新类标签

        if (abs(new_loss - old_loss) < GMM_epsilon):  # 小于误差值，结束循环
            break

    argmax = np.argmax(gamma, axis=1)
    num = np.zeros(k)

    for j in range(N):
        label = argmax[j]
        category[j] = label  # 更新类标签
        num[label] += 1
    return category, num, count


def GMM_show(k, n, dim, mu, sigma):
    "GMM算法的结果展示"

    data = gen_data(k, n, dim, mu, sigma)
    # print(data.shape)  # (1200,2)
    N = data.shape[0]  # N为样本点总数

    category, num, count = GMM(data, k, N, dim, mu, sigma)

    acc = calc_acc(num, n, N)
    print("acc:" + str(acc))

    for i in range(N):  # 绘制分为各类的样本点
        color_num = int(category[i] % len(colors))
        plt.scatter(data[i, 0], data[i, 1],
                    c=colors[color_num], s=5)

    plt.title("GMM results " + "   config:" + str(con) + "   iter:" + str(count))
    plt.show()
    return


def UCI_read():
    "读入UCI数据并切分"
    raw_data = pd.read_csv("./iris.csv")
    data = raw_data.values
    label = data[:, -1]
    np.delete(data, -1, axis=1)
    # print(data.shape) # (150, 5)
    return data, label


def UCI_show():
    "使用UCI数据集测试算法"
    data, label = UCI_read()

    k = 3  # 聚类数
    N = data.shape[0]  # 样本数量
    n = N / k  # 每一类样本数量
    dim = data.shape[1]  # 样本维度

    # K-means算法计算结果
    category, center, iter_count, num = kmeans(data, k, N, dim)
    kmeans_acc = calc_acc(num, n, N)
    print("K-means算法结果：")
    print("迭代数：%d   准确率:%.2f\n" % (iter_count, kmeans_acc))

    # GMM算法计算结果
    GMM_mu = np.array([[0] * dim] * k)
    GMM_sigma = np.array([np.eye(dim)] * k)
    temp_counts = np.array([0] * k)
    temp = np.array([[0] * dim] * k)
    for i in range(N):
        c = int(category[i, -1])
        temp_counts[c] += 1
        for j in range(dim):
            temp[c, j] += category[i, j]

    for i in range(k):
        for j in range(dim):
            GMM_mu[i, j] = temp[i, j] / temp_counts[i]

    for i in range(k):
        for j in range(dim):
            for t in range(N):
                if category[i, -1] == i:
                    GMM_sigma[i, j,
                              j] += pow((category[i, j] - GMM_mu[i, j]), 2)
            GMM_sigma[i, j, j] /= temp_counts[i]

    category, num, count = GMM(data, k, N, dim, GMM_mu, GMM_sigma)
    GMM_acc = calc_acc(num, n, N)
    print("GMM算法结果：")
    print("迭代数：%d   准确率:%.2f\n" % (count, GMM_acc))
    return


# 主函数
config = configs[con]  # 选取配置
k, n, dim, mu, sigma = config['k'], config['n'], config['dim'], config['mu'], config['sigma']

if method == 0:
    kmeans_show(k, n, dim, mu, sigma)
elif method == 1:
    GMM_show(k, n, dim, mu, sigma)
elif method == 2:
    UCI_show()
