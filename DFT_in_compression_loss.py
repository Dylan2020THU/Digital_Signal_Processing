# Discrete Fourier Transform
# 2021-5-11
# TBSI, THU
# ZHX

from matplotlib import pyplot as plt
import numpy as np
import math


def DFT(input, K, N):
    """
    :param input: the input signal
    :param K: the number of basis signals
    :param N: the number of sampling points
    """

    # DFT
    X = np.zeros(K)
    Re_X = np.zeros(K)
    Im_X = np.zeros(K)
    # print('X',X)

    for k in range(K):
        for n in range(N):
            Re_X[k] += input[n] * np.cos(2 * np.pi * (n * k / N))
            Im_X[k] += input[n] * np.sin(2 * np.pi * (n * k / N))
        X[k] = np.sqrt(Re_X[k]**2 + Im_X[k]**2)
    return X


def cal_cmp_loss(term_num, N, X_k):

    X_sum = sum(X_k)
    for k in range(term_num):
        if k <= math.floor(term_num / 2):
            X_sum -= X_k[k]
        else:
            X_sum -= X_k[N-term_num+k]
    compression_loss = X_sum / N
    return compression_loss


if __name__ == '__main__':
    n_start = 0
    n_end = 32
    num = n_end - n_start
    n = np.arange(n_start, n_end)
    print(n)
    x_n = np.exp(-n / 10)  # input signal

    #  Draw the curve of the compression loss using MSE
    L = num  # the first L terms in DFT coefficients
    # L = L+1

    X_k = DFT(x_n, L, num)

    tmp_loss = np.zeros(L)
    for l in range(L):
        if l % 2 == 0:  # odd only
            tmp_loss[l] = cal_cmp_loss(l, num, X_k)
            print('compression_loss[%s]'%l, tmp_loss[l])

    # delete zero in array loss
    loss = []
    for l in range(L):
        if tmp_loss[l] != 0:
            loss.append(tmp_loss[l])

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set(xlabel='L', ylabel='loss')
    ax2.scatter(2*np.arange(len(loss)), loss, c='b', label='DFT loss')

    plt.title('DFT')
    plt.legend()
    plt.show()
