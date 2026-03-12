import numpy as np
import matplotlib.pyplot as plt

# sizes for HW07 sweep
n = np.array([2**i for i in range(10,21)])

thrust = np.loadtxt("times_thrust.txt")
cub = np.loadtxt("times_cub.txt")

# HW05 data
hw05_256 = np.loadtxt("times256.txt")
hw05_1024 = np.loadtxt("times1024.txt")

n_hw05 = np.array([2**i for i in range(10,21)])

plt.figure()

plt.loglog(n, thrust, 'o-', label="Thrust Reduction")
plt.loglog(n, cub, 's-', label="CUB Reduction")

plt.loglog(n_hw05, hw05_256, '^-', label="HW05 Reduction (256 threads)")
plt.loglog(n_hw05, hw05_1024, 'd-', label="HW05 Reduction (1024 threads)")

plt.xlabel("Input size (n)")
plt.ylabel("Time (ms)")
plt.title("Reduction Scaling Comparison")
plt.legend()
plt.grid(True)

plt.savefig("task1.pdf")
