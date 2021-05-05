from matplotlib import pyplot as plt
import numpy as np

t_start = 0
t_end = 1
num_of_point = 50

t = np.linspace(t_start, t_end, num=num_of_point)
print(t)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set(xlabel='t', ylabel='phi_k_(t)')

phi_1_t = np.ones(num_of_point)
ax1.plot(t, phi_1_t, label='k=0')

for k in range(1, 5):
    phi_k_t = np.sqrt(2) * np.cos(np.pi * k * t)
    ax1.plot(t, phi_k_t, label='k=%s' % k)

plt.legend()
plt.show()
