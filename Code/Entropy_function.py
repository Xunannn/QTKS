from scipy.signal import resample
import math
# 样本熵函数实现
def sample_entropy(dim, r, data, tau=1):
    if tau > 1:
        data = resample(data, len(data) // tau)

    N = len(data)
    result = np.zeros(2)

    for m in range(dim, dim + 2):
        Bi = np.zeros(N - m + 1)
        data_mat = np.zeros((N - m + 1, m))

        # 构造数据矩阵，形成 m 维向量
        for i in range(N - m + 1):
            data_mat[i, :] = data[i:i + m]

        # 利用距离计算相似模式数
        for j in range(N - m + 1):
            # 计算切比雪夫距离，不包括自匹配情况
            dist = np.max(np.abs(data_mat - np.tile(data_mat[j, :], (N - m + 1, 1))), axis=1)
            # 统计 dist 小于等于 r 的数量，不包括自匹配情况
            D = (dist <= r)
            Bi[j] = (np.sum(D) - 1) / (N - m)

        # 求所有 Bi 的均值
        result[m - dim] = np.sum(Bi) / (N - m + 1)

    # 计算得到的样本熵值
    samp_en = -np.log(result[1] / result[0])
    return samp_en

# 排列熵函数实现
import numpy as np
from math import factorial

def Permutation_Entropy(time_series, order, delay, normalize):
    # 将时间序列转换为NumPy数组
    x = np.array(time_series)  # x = [4, 7, 9, 10, 6, 11, 3]

    # 生成权重数组，用于编码排序后的排列
    hashmult = np.power(order, np.arange(order))  # [1 3 9]

    # 生成排序后的索引矩阵
    # _embed的作用是生成嵌入后的时间序列矩阵
    # argsort的作用是对下标排序，排序的标准是值的大小
    # 比如第3行[9,10,6]，9的下标是0，6是2，排序后向量的第一个元素是6的下标2，排完[201]
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')

    # 计算哈希值
    # np.multiply 对应位置相乘，hashmult是1 3 9，sum是求每一行的和
    # hashmult一定要保证三个一样的值顺序不同，按位乘起来后每一行加起来大小不同，类似赋一个权重
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)  # [21 21 11 19 11]

    # 统计每个哈希值的出现次数
    # np.unique排序并返回唯一值，c是每个数字出现的次数，最小的11出现了2次，19 1次
    _, c = np.unique(hashval, return_counts=True)

    # 计算概率分布
    p = np.true_divide(c, c.sum())  # [0.4 0.2 0.4]，2/5=0.4

    # 计算排列熵
    pe = -np.multiply(p, np.log2(p)).sum()  # 根据公式

    # 归一化（如果需要）
    if normalize:
        pe /= np.log2(factorial(order))

    return pe


# 将一维时间序列，生成矩阵
def _embed(x, order=3, delay=1):

    N = len(x)
    Y = np.empty((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T

# 近似熵函数实现
def k_approximate_entropy(time_series, m, r):

    time_series = np.squeeze(time_series)

    def max_dist(x_i, x_j):
        # 计算两个数据运行之间的最大距离
        return max([abs(ia - ja) for ia, ja in zip(x_i, x_j)])

    def phi(m):
        # 计算 Phi 值
        x = [[time_series[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        # 对每个子序列进行比较，并计算符合条件的数量
        C = [len([1 for x_j in x if max_dist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        # 计算 Phi 值
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(time_series)

    # 返回近似熵的差值
    return phi(m) - phi(m + 1)


# 模糊熵函数函数实现

def fuzzy_entropy(data, dim, r, n=2, tau=1):
    if tau > 1:
        data = data[::tau]

    N = len(data)
    result = np.zeros(2)

    for m in range(dim, dim + 2):
        count = np.zeros(N - m + 1)
        data_mat = np.zeros((N - m + 1, m))

        for i in range(N - m + 1):
            data_mat[i, :] = data[i:i + m]

        for j in range(N - m + 1):
            data_mat = data_mat - np.mean(data_mat, axis=1, keepdims=True)
            temp_mat = np.tile(data_mat[j, :], (N - m + 1, 1))
            dist = np.max(np.abs(data_mat - temp_mat), axis=1)
            D = np.exp(-(dist ** n) / r)
            count[j] = (np.sum(D) - 1) / (N - m)

        result[m - dim] = np.sum(count) / (N - m + 1)

    fuz_en = np.log(result[0] / result[1])
    return fuz_en

# 信息熵
def info_entropy(data):
    """
    计算信息熵
    :param data: 数据集
    :return: 信息熵
    """
    length = len(data)
    counter = {}
    for item in data:
        counter[item] = counter.get(item, 0) + 1
    ent = 0.0
    for _, cnt in counter.items():
        p = float(cnt) / length
        ent -= p * math.log2(p)
    return ent