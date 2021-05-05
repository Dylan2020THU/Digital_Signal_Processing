# Discrete Cosine Transform
# 2021-5-5
# TBSI, THU
# ZHX

from matplotlib import pyplot as plt
import numpy as np

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set(xlabel='t', ylabel='phi_k_(t)')

# The cosine-I basis functions for k-dimensional original signals on [0,1]
t_start = 0
t_end = 1
num_of_point = 50

t = np.linspace(t_start, t_end, num=num_of_point)
print('t:', t)

# phi = 1, for k = 0
phi_0_t = np.ones(num_of_point)
ax1.plot(t, phi_0_t, label='k=0')

# phi = sqrt(2) * cos(pi * k * t), for k = 1,2...
for k in range(1, 5):
    phi_k_t = np.sqrt(2) * np.cos(np.pi * k * t)
    ax1.plot(t, phi_k_t, label='k=%s' % k)
plt.legend()

# The DCT cosine-I basis functions for N-dimensional original signals

ax2 = fig.add_subplot(212)
ax2.set(xlabel='n', ylabel='phi_k_(n)')

N = 5
n = np.linspace(0, N, num=N, dtype=int)
print('n:', n)

# phi = sqrt(1/N), for k = 0
phi_0_n = np.sqrt(2 / N) * np.ones(N)
ax2.plot(n, phi_0_n, label='k=0')

# phi = sqrt(2/N) * cos((pi * k)/N *(n+1/2), for k = 1,2...
for k in range(1, N):
    phi_k_n = np.sqrt(2 / N) * np.cos((np.pi * k) / N * (n + 1 / 2))
    ax2.plot(n, phi_k_n, label='k=%s' % k)

plt.legend()
plt.show()
