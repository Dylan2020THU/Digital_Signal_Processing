# Tikhonov Regularization
# 2021-5-16
# TBSI, THU
# Hengxi Zhang

import numpy as np
from scipy import io as io
from matplotlib import pyplot as plt

if __name__ == '__main__':

    matr = io.loadmat(r'F:\清华大学TBSI\2 Spring 2021\1 Courses\Advanced Signal Processing\ASP_HW\ASP_hw7\blocks.mat')
    x = matr['x'].reshape(1,-1)  # signal
    h = matr['h'].reshape(1,-1)  # filter
    y = matr['y'].reshape(1,-1)  # convolution
    yn = matr['yn'].reshape(1,-1)  # noisy observation of y

    # ----------------- 7.1 (a) Construct Transition Matrix -----------------#
    M = y.size
    N = x.size
    h_len = h.size

    A = np.zeros((M, N))  # Apply for the storage space for matrix A
    for i in range(N):
        A[i:i + h_len, i] = h  # Construct A
    print(A)

    #----------------- 7.2 (b) Find the reconstruction error of pseudo inverse solution -----------------#

    A_pinv = np.linalg.pinv(A)  # pseudo inverse
    x_pinv = np.dot(A_pinv, yn.reshape(-1,1))
    recon_error_pinv = np.linalg.norm(x_pinv - x.reshape(-1,1))  # Calculate the error

    #----------------- 7.3 (c) Find the reconstruction error of truncated SVD solution -----------------#

    U, sigma, V_T = np.linalg.svd(A)  # SVD of A
    V = V_T.T  # Transpose
    U_T = U.T
    ''''''
    recon_error_trun = np.zeros(N)
    for trun_num in range(N):  # # of the truncation
        A_trun_pinv = np.zeros((N, M))
        for r in range(trun_num):
            A_trun_pinv = A_trun_pinv + np.dot(V[:, r].reshape((-1, 1)), U_T[r, :].reshape((1, -1))) / sigma[r]  # SVD
            # print('# %s :' % r, A_trun_pinv)
        x_trun = np.dot(A_trun_pinv, yn.reshape(-1,1))  # Calculate the solution x_trun
        recon_error_trun[trun_num] = np.linalg.norm(x_trun.reshape(-1,1) - x.reshape(-1,1))  # Calculate the error

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set(xlabel='num', ylabel='error_trun')
    ax1.plot(np.arange(N), recon_error_trun, c='r', label='error')
    plt.title('Reconstruction Error of Truncated SVD Solution')
    plt.legend()


    #----------------- 7.4 (d) Find the reconstruction error of Tikhonov regularization solution -----------------#
    delta_start = 0.0001  # Set the range of delta
    delta_end = 0.1
    delta = np.linspace(delta_start, delta_end, num=10)
    delta_num = len(delta)
    recon_error_Tik = np.zeros(delta_num)  # Apply for the storage space
    para_Tik = np.zeros((delta_num, N))

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(211)
    plt.title('sigma - para_Tik')
    ax1.set(xlabel='sigma', ylabel='sigma / sigma^2 + delta')

    x_Tik = np.zeros((delta_num, N))
    gap = np.zeros((delta_num, N))
    for i in range(delta_num):
        for r in range(N):
            para_Tik[i][r] = sigma[r] / (sigma[r] ** 2 + delta[i])  # parameter of Tikhonov
            x_Tik[i] = x_Tik[i] + para_Tik[i][r] * np.dot(yn, U[:, r]) * V[:, r]  # Calculate the solution x_trun using Tikhonov
        recon_error_Tik[i] = np.linalg.norm(x_Tik[i].reshape(-1,1) - x.reshape(-1,1))  # Calculate the error

        ax1.plot(sigma, para_Tik[i], label='delta = %.2f' % delta[i])
        plt.legend()

    ax2 = fig2.add_subplot(212)
    plt.title('Reconstruction Error of Tikhonov regularization solution')
    ax2.set(xlabel='# of delta', ylabel='error_Tikhonov')
    ax2.plot(delta, recon_error_Tik, label='error')
    plt.legend()

    plt.show()
