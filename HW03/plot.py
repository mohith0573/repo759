import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("timing.txt", skiprows=1)

n = data[:,0]
t512 = data[:,1]
t16 = data[:,2]

plt.figure()
plt.plot(n, t512, label="512 threads/block")
plt.plot(n, t16, label="16 threads/block")

plt.xlabel("n")
plt.ylabel("Time (ms)")
plt.legend()
plt.grid(True)

plt.savefig("task3.pdf")

