import numpy as np
import matplotlib.pyplot as plt

times = np.loadtxt("times_task3_t.txt")
t = np.arange(1, len(times)+1)

plt.plot(t, times, 'o-')
plt.xlabel("Threads (t)")
plt.ylabel("Time (ms)")
plt.title("HW08 Task3: Time vs Threads")
plt.grid()

plt.savefig("hw8_task3_t.pdf")
