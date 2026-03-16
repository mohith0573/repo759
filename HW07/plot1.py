import numpy as np
import matplotlib.pyplot as plt

# HW07 sweep results
thrust = np.loadtxt("times_thrust.txt")
cub = np.loadtxt("times_cub.txt")

n_hw07 = np.array([2**i for i in range(10,21)])

# HW05 raw files
raw256 = np.loadtxt("times256.txt")
raw1024 = np.loadtxt("times1024.txt")

# Step 1: take only timing values
times256 = raw256[1::2]
times1024 = raw1024[1::2]

# Step 2: take every other timing value (match HW07 sweep points)
times256 = times256[::2]
times1024 = times1024[::2]

n_hw05 = np.array([2**i for i in range(10,21)])

plt.figure()

plt.loglog(n_hw07, thrust, 'o-', label="Thrust")
plt.loglog(n_hw07, cub, 's-', label="CUB")

plt.loglog(n_hw05, times256, '^-', label="HW05 CUDA (256 threads)")
plt.loglog(n_hw05, times1024, 'd-', label="HW05 CUDA (1024 threads)")

plt.xlabel("Input size (n)")
plt.ylabel("Time (ms)")
plt.title("Reduction Scaling Comparison")

plt.grid(True)
plt.legend()

plt.savefig("task1.pdf")
