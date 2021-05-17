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
    plt.show()