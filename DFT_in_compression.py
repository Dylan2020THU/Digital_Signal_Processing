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
    X = np.array(K)
    for k in range(K):
        X[k] = complex(0,0)
    # Re_X = np.zeros(K)
    # Im_X = np.zeros(K)
    # print('X',X)

    for k in range(K):
        for n in range(N):
            X[k] += input[n] * complex(np.cos(2 * np.pi * (n * k / num)), np.sin(2 * np.pi * (n * k / num)))
            # Re_X[k] += input[n] * np.cos(2 * np.pi * (n * k / N))
            # Im_X[k] += input[n] * np.sin(2 * np.pi * (n * k / N))
        # X[k] = np.sqrt(Re_X[k]**2 + Im_X[k]**2)
    return X


def fit_curve_DFT(n, term_num):
    # DCT
    # term_num = 6
    X_k = DFT(x_n, term_num, num)  # Calculate each X[k]
    print('X[k]', X_k)
    # Draw the fig of input signal
    global fig
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set(xlabel='n', ylabel='x[n]')
    ax1.plot(n, x_n, label='x[n]')  # draw x[n]
    ax1.scatter(np.arange(term_num), X_k, c='r')  # draw X[k]

    x_head = np.zeros((term_num, num))  # Apply for r.v. space
    x_head_total = np.zeros(num)
    for k in range(term_num):
        if k <= math.ceil(term_num / 2 - 1):
            for i in n:
                x_head[k][i] = X_k[k] * np.cos(2 * np.pi * (i * k / num))
        else:
            for i in n:
                x_head[k][i] = X_k[term_num-k] * complex(np.cos(2 * np.pi * (i * (term_num-k) / num)), np.sin(2 * np.pi * (i * (term_num-k) / num)))  # ???
        # print('x_head',x_head)
        x_head_total += x_head[k]
        ax1.plot(n, x_head[k], c='g', label='basis_%s[n]' % k)

    x_head_total = x_head_total / num

    # print('x_head:', x_head)
    ax1.plot(n, x_head_total, label='x_fit_curve[n]')
    plt.legend()
    plt.title('DFT')

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
    L = 5  # the first L terms in DFT coefficients
    L = L+1
    loss = np.zeros(L)
    for l in range(L):
        loss[l] = fit_curve_DFT(n, l)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set(xlabel='L', ylabel='loss')
    ax2.plot(np.arange(L), loss, c='b', label='loss')

    plt.title('DFT')
    plt.legend()
    plt.show()
