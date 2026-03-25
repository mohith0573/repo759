import matplotlib.pyplot as plt

sizes = [512,1024,2048,4096]
times = []

with open("task4_data.txt") as f:
    for line in f:
        times.append(float(line.strip()))

plt.plot(sizes, times, marker='o')
plt.xlabel("Matrix Size")
plt.ylabel("Time (ms)")
plt.grid()
plt.savefig("task4.pdf")
