# Discrete Fourier Transform
# 2021-5-7
# TBSI, THU
# ZHX

from matplotlib import pyplot as plt
import numpy as np


def DFT(input, K, N):
    """
    :param input: the input signal
    :param K: the number of basis signals
    :param N: the number of sampling points
    """

    # DFT
    X = np.zeros(K)
    for k in range(K):
        for n in range(N):
            X[k] += input[n] * np.cos(2 * np.pi * k * n / N)
        print('X[%s]=' % k, X[k])
        ax1.scatter(k, X[k])


def draw_basis_funcions(start, end, K, N):
    samples = 100
    n = np.linspace(start, end, num=samples)
    basis = np.zeros((K, samples))
    for k in range(K):
        basis[k] = np.cos(2 * np.pi * k * n / N)
        ax1.plot(n, basis[k], label='basis[%s]' % k)


if __name__ == '__main__':
    n_start = 0
    n_end = 2 * np.pi
    n = np.linspace(n_start, n_end)
    x_n = np.cos(n)  # input signal

    # Draw the fig of input signal
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set(xlabel='n', ylabel='x[n]')
    ax1.plot(n, x_n, label='x[n]')  # input signals

    draw_basis_funcions(n_start, n_end, 3, 10)  # basis functions
    DFT(x_n, 4, 10)  # DFT

    plt.legend()
    plt.show()
