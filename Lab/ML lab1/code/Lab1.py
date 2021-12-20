import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline

# 全局变量
sigma = 0.1  # 生成数据的标准差
data_size = 100  # 数据点数量
lambda_ML = 1e-11  # 超参数 λ
m = 50  # 拟合多项式的阶数
learning_rate = 1e-6  # 学习率
epsilon = 1e-5 # 允许的误差值


# 产生带0均值高斯噪声的数据
def generate_data():
    x = np.random.random(data_size)
    x = np.sort(x)
    noise = np.random.normal(0, sigma, data_size)
    y = np.sin(2 * np.pi * x) + noise
    x = x.reshape(data_size, 1)
    y = y.reshape(data_size, 1)
    return x, y


# 构造x的Vandermonde矩阵（横着的）
def generate_vandermonde(x):
    van = np.ones((x.shape[0], 1))
    for i in range(1, m + 1):
        van = np.hstack((van, x**i))
    return van


# 最小二乘法求解w
def least_square(van, y):
    w = np.linalg.inv((van.T @ van)) @ van.T @ y
    return w


# 带正则项的最小二乘法求解w
def least_square_regular(van, y):
    w = np.linalg.inv(
        (van.T @ van + np.eye(van.T.shape[0]) * lambda_ML)) @ van.T @ y
    return w


# 确定最佳的超参数λ
def RMS(van, y):
    ln_lambda = np.linspace(0, 10, 101).reshape(1, 101).T
    E_RMS = np.empty((1, 101))
    for i in range(0, 101):
        real_lambda = np.log10(1.0 / ln_lambda[i])
        w = np.linalg.inv(
            van.T @ van + np.eye(van.T.shape[0]) * real_lambda) @ van.T @ y
        Ew = 0.5 * (van @ w - y).T @ (van @ w - y)
        E_RMS[0, i] = (np.sqrt(2 * Ew / data_size))
    E_RMS = E_RMS.T
    # 绘图
    plt.figure(2)
    plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.plot(ln_lambda, E_RMS, c='orange', linewidth=2,
             label='E_RMS')
    plt.xlabel("10^(-x)")
    plt.ylabel("E_RMS")
    plt.legend()
    plt.show()
    return 0


# 损失函数
def loss(van, y, w):
    diff = van @ w - y
    loss = 0.5 * diff.T @ diff
    return loss


# 梯度函数
def gradient_function(van, y, w):
    grad = van.T @ van @ w - van.T @ y + lambda_ML * w
    return grad


# 梯度下降法
def gradient_descent(van, y, learning_rate):
    w = np.zeros((m + 1, 1))
    grad = gradient_function(van, y, w)
    loss0 = 0
    loss1 = loss(van, y, w)
    count = 0
    while abs(loss1 - loss0) > epsilon:  # 误差值，小于该值时停止迭代
        count += 1
        w = w - learning_rate * grad
        loss0 = loss1
        loss1 = loss(van, y, w)
        if(loss1 > loss0):  # loss不降反增，则减半学习率
            learning_rate *= 0.5
        grad = gradient_function(van, y, w)
        print(count)
    return w, count


# 共轭梯度法
def conjugate_gradient(van, y):
    c_lambda = 1e-4
    A = van.T @ van + c_lambda
    b = van.T @ y
    w = np.zeros((van.shape[1], 1))
    r = b
    p = b
    count = 0
    while True:
        count += 1
        if r.T @ r < epsilon:
            break
        norm = r.T @ r
        a = norm / (p.T @ A @ p)
        w = w + a * p
        r = r - (a * A @ p)
        b = (r.T @ r) / norm
        p = r + b * p
    return w, count

# 根据已经求得的w计算实际的 x, y 坐标值


def fitting(van, w):
    xw = van[:, 1]
    yw = van @ w
    return xw, yw


x, y = generate_data()
# print(x)
# print(y.shape)
van = generate_vandermonde(x)
# print(van)
# print(van.shape)

method = ['least_square', 'least_square_regular',
          'gradient_descent', 'conjugate_gradient']

choose_method = method[2]  # 选择方法

count = 0

if choose_method == "least_square":
    w = least_square(van, y)
elif choose_method == "least_square_regular":
    w = least_square_regular(van, y)
elif choose_method == "gradient_descent":
    w, count = gradient_descent(van, y, learning_rate)
elif choose_method == "conjugate_gradient":
    w, count = conjugate_gradient(van, y)
else:
    w = 0
# print(w)
# print(w.shape)

xw, yw = fitting(van, w)
# print(xw.shape)
# print(yw.shape)


# 绘图部分
plt.xlim(0, 1)
plt.scatter(x, y, c='grey', s=10, label='data point')  # 生成的数据点
temp = np.linspace(0, 1, 10000)
plt.plot(temp, np.sin(2 * np.pi * temp), c='orange',
         linewidth=2, label='y = sin(2πx)')  # 标准的 y = sin(2πx) 函数
xw_smooth = np.linspace(xw.min(), xw.max(), 300)
yw_smooth = make_interp_spline(xw, yw)(xw_smooth)
plt.plot(xw_smooth, yw_smooth, c='blue', linewidth=2,
         label='fitting')  # 拟合得到的函数（进行了平滑）
# plt.title('m = ' + str(m) + ',' + 'data_size = ' + str(data_size))
# plt.title('m = ' + str(m) + ',' + 'data_size = ' + str(data_size) + ',' + 'λ = ' + str(lambda_ML)) # 解析解（带正则项）
# plt.title('m = ' + str(m) + ',' + 'data_size = ' + str(data_size) + ',' + 'υ = ' + str(epsilon) + ',' + 'learning_rate = ' + str(learning_rate) + ',' + 'count = ' + str(count) )  # 梯度下降
plt.title('m = ' + str(m) + ',' + 'data_size = ' + str(data_size) + ',' + 'count = ' + str(count) )  # 共轭梯度
plt.legend()
plt.show()
