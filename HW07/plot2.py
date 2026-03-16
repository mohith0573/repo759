import numpy as np
import matplotlib.pyplot as plt

times = np.loadtxt("times_task2.txt")

n = np.array([2**i for i in range(5,21)])

plt.figure()

plt.loglog(n, times, 'o-')

plt.xlabel("Input size (n)")
plt.ylabel("Time (ms)")
plt.title("Count Kernel Scaling")
plt.grid(True)

plt.savefig("task2.pdf")
