import matplotlib.pyplot as plt
import numpy as np

times = []
sizes = []

with open("task3_data.txt") as f:
    for i, line in enumerate(f):
        times.append(float(line.strip()))
        sizes.append(2**(i+1))

plt.loglog(sizes, times, marker='o')
plt.xlabel("Message Size (n)")
plt.ylabel("Time (ms)")
plt.grid()
plt.savefig("task3.pdf")
