import numpy as np
import matplotlib.pyplot as plt

times = np.loadtxt("times_task2_hw8.txt")
t = np.arange(1, len(times)+1)

plt.plot(t, times, 'o-')
plt.xlabel("Threads (t)")
plt.ylabel("Time (ms)")
plt.title("HW08 Task2: Convolution Scaling")
plt.grid()

plt.savefig("hw8_task2.pdf")
