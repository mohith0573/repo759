import numpy as np
import matplotlib.pyplot as plt

# load times
times = np.loadtxt("times_task1.txt")

# n values
n = np.array([2**i for i in range(5,12)])

plt.figure(figsize=(8,5))

plt.plot(n, times, 'o-', linewidth=2)

plt.xlabel("Matrix size n")
plt.ylabel("Time (ms)")
plt.title("cuBLAS Matrix Multiplication Performance")

plt.xscale("log", base=2)
plt.yscale("log")

plt.grid(True, which="both", linestyle="--")
plt.tight_layout()

plt.savefig("task1.pdf")
