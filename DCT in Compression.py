# Discrete Fourier Transform
# 2021-5-10
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

    for k in range(K):
        # phi = sqrt(1/N), for k = 0
        if k == 0:
            for n in range(N):
                X[k] += input[n] * np.sqrt(1/N)

        # phi = sqrt(2/N) * cos((pi * k)/N *(n+1/2), for k = 1,2,...
        else:
            for n in range(N):
                X[k] += input[n] * np.cos(2 * np.pi * k * n / N)
    return X


if __name__ == '__main__':
    n_start = 0
    n_end = 32
    num = n_end - n_start
    n = np.arange(n_start, n_end)
    print(n)
    x_n = np.exp(-n/10)  # input signal

    # DCT
    term_num = 5
    X_k = DCT(x_n, term_num, num)

    # Draw the fig of input signal
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set(xlabel='n', ylabel='x[n]')
    ax1.plot(n, x_n, label='x[n]')  # draw x[n]
    ax1.scatter(np.arange(term_num), X_k, c='r')  # draw X[k]

    plt.legend()
    plt.show()