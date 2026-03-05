import numpy as np
import matplotlib.pyplot as plt

times = []

with open("times_task2.txt") as f:
    lines = f.readlines()

for i in range(1, len(lines), 2):
    times.append(float(lines[i].strip()))

n = [2**i for i in range(10,17)]

plt.figure(figsize=(8,5))

plt.plot(n, times, 'o-', linewidth=2)

plt.xlabel("Array size n")
plt.ylabel("Time (ms)")
plt.title("Hillis-Steele Scan Performance")

plt.xscale("log", base=2)
plt.yscale("log")

plt.grid(True, which="both", linestyle="--")
plt.tight_layout()

plt.savefig("task2.pdf")
