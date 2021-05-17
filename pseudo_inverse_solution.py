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
    print(recon_error_pinv)
