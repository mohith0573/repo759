import matplotlib.pyplot as plt

times = []
threads = list(range(1, 11))

with open("task1_data.txt") as f:
    lines = f.readlines()
    for i in range(2, len(lines), 3):
        times.append(float(lines[i]))

plt.plot(threads, times, marker='o')
plt.xlabel("Threads")
plt.ylabel("Time (ms)")
plt.title("Task1 Performance")
plt.grid()
plt.savefig("task1.pdf")
