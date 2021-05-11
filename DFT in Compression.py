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
    # print('X',X)

    for k in range(K):
        for n in range(N):
            X[k] += input[n] * np.cos(2 * np.pi * (n * k / N))

    return X


def fit_curve_DFT(term_num):
    # DCT
    # term_num = 6
    X_k = DFT(x_n, term_num, num)  # Calculate each X[k]

    # Draw the fig of input signal
    global fig
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.set(xlabel='n', ylabel='x[n]')
    ax1.plot(n, x_n, label='x[n]')  # draw x[n]
    ax1.scatter(np.arange(term_num), X_k, c='r')  # draw X[k]

    x_head = np.zeros((term_num, num))  # Apply for r.v. space
    x_head_total = np.zeros(num)
    for k in range(term_num):
        if k <= math.ceil(term_num / 2 - 1):
            x_head[k] = X_k[k] * np.cos(2 * np.pi * (n * k / num))
        else:
            x_head[k] = X_k[num-k] * np.cos(2 * np.pi * (n * k / num))
        x_head_total += x_head[k]
        ax1.plot(n, x_head[k], c='g', label='basis_%s[n]' % k)

    x_head_total = x_head_total / num

    # print('x_head:', x_head)
    ax1.plot(n, x_head_total, label='x_head[n]')

    compression_loss = sum((x_head_total - x_n) * (x_head_total - x_n)) / num
    return compression_loss


if __name__ == '__main__':
    n_start = 0
    n_end = 32
    num = n_end - n_start
    n = np.arange(n_start, n_end)
    print(n)
    x_n = np.exp(-n / 10)  # input signal

    #  Draw the curve of the compression loss using MSE
    L = num
    loss = np.zeros(L)
    for l in range(L):
        loss[l] = fit_curve_DFT(l)

    ax2 = fig.add_subplot(212)
    ax2.plot(np.arange(L), loss, c='b')

    plt.legend()
    plt.show()
