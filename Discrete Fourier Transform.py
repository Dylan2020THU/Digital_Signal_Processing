# Discrete Fourier Transform
# 2021-5-7
# TBSI, THU
# ZHX

from matplotlib import pyplot as plt
import numpy as np


def DFT(input,K):
    """
    :param input: the input signal
    :param K: the number of sampling points
    """

    # DFT
    X = np.zeros(K)
    N = len(input)
    for k in range(K):
        for n in range(N-1):
            X[k] += input[n]*np.cos(2*np.pi*k*n/N)
        ax1.scatter(k, X[k])


if __name__ == '__main__':

    N = 2*np.pi  # Sampling points
    n = np.linspace(0, N, num=100)
    x_n = np.cos(n)

    # Draw the fig of input signal
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set(xlabel='n', ylabel='x[n]')
    ax1.plot(n, x_n, label='x[n]')

    DFT(x_n, 10)

    plt.legend()
    plt.show()
