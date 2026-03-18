import numpy as np
import matplotlib.pyplot as plt

times = np.loadtxt("times_task3_ts.txt")
ts = [2**i for i in range(1, len(times)+1)]

plt.semilogx(ts, times, 'o-')
plt.xlabel("Threshold (ts)")
plt.ylabel("Time (ms)")
plt.title("HW08 Task3: Time vs Threshold")
plt.grid()

plt.savefig("hw8_task3_ts.pdf")
