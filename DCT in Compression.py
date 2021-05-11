# Discrete Fourier Transform
# 2021-5-11
# TBSI, THU
# ZHX

from matplotlib import pyplot as plt
import numpy as np


def DCT(input, K, N):
    """
    :param input: the input signal
    :param K: the number of basis signals
    :param N: the number of sampling points
    """

    # DFT
    X = np.zeros(K)
    # print('X',X)

    for k in range(K):
        # phi = sqrt(1/N), for k = 0
        if k == 0:
            for n in range(N):
                X[k] += input[n] * np.sqrt(1 / N)

        # phi = sqrt(2/N) * cos((pi * k)/N *(n+1/2), for k = 1,2,...
        else:
            for n in range(N):
                X[k] += input[n] * np.sqrt(2 / N) * np.cos((np.pi * k) / N * (n + 0.5))
    return X


def fit_curve_DCT(term_num):
    # DCT
    # term_num = 6
    X_k = DCT(x_n, term_num, num)  # Calculate each X[k]

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
        if k == 0:
            x_head[k] = X_k[k] * np.sqrt(1 / num)
        else:
            x_head[k] = X_k[k] * np.sqrt(2 / num) * np.cos((np.pi * k) / num * (n + 0.5))
        x_head_total += x_head[k]
        ax1.plot(n, x_head[k], c='g', label='basis_%s[n]' % k)

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
        loss[l] = fit_curve_DCT(l)

    ax2 = fig.add_subplot(212)
    ax2.plot(np.arange(L), loss, c='r')

    plt.legend()
    plt.show()
