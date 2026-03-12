import numpy as np
import matplotlib.pyplot as plt

# HW07 sweep data
thrust = np.loadtxt("times_thrust.txt")
cub = np.loadtxt("times_cub.txt")

n_hw07 = np.array([2**i for i in range(10,21)])

# HW05 files contain alternating result/time
hw05_256_raw = np.loadtxt("times256.txt")
hw05_1024_raw = np.loadtxt("times1024.txt")

# extract only timing values
hw05_256 = hw05_256_raw[1::2]
hw05_1024 = hw05_1024_raw[1::2]

n_hw05 = np.array([2**i for i in range(10,21)])

plt.figure()

plt.loglog(n_hw07, thrust, 'o-', label="Thrust")
plt.loglog(n_hw07, cub, 's-', label="CUB")

plt.loglog(n_hw05, hw05_256, '^-', label="HW05 CUDA (256 threads)")
plt.loglog(n_hw05, hw05_1024, 'd-', label="HW05 CUDA (1024 threads)")

plt.xlabel("Input size (n)")
plt.ylabel("Time (ms)")
plt.title("Reduction Scaling Comparison")

plt.legend()
plt.grid(True)

plt.savefig("task1.pdf")
