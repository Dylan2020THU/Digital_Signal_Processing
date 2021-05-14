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
    X = np.zeros(N)
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


def cal_cmp_loss(X_k, term_num, num):
    # calculate the compression loss equivalent to the sum of the unused DCT coefficients
    compression_loss = 0
    for k in range(1, num-term_num+1):
        compression_loss += X_k[-k]
    return compression_loss


if __name__ == '__main__':
    n_start = 0
    n_end = 32
    num = n_end - n_start
    n = np.arange(n_start, n_end)
    print(n)
    x_n = np.exp(-n / 10)  # input signal

    # Calculate the loss
    L = num
    X_k = DCT(x_n, L, num)
    loss = np.zeros(L)
    for l in range(L):
        loss[l] = cal_cmp_loss(X_k, l, num)
        print('compression_loss[%s]' % l, loss[l])


    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(np.arange(len(loss)), loss, c='r', label='DCT loss')

    plt.title('DCT')
    plt.legend()
    plt.show()
