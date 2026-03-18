import numpy as np
import matplotlib.pyplot as plt

t = np.arange(1,21)
times = np.loadtxt("times_task1.txt")

plt.plot(t, times, 'o-')
plt.xlabel("Threads")
plt.ylabel("Time (ms)")
plt.title("Matrix Multiplication Scaling")

plt.savefig("hw8_task1.pdf")
