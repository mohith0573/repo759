import matplotlib.pyplot as plt

threads = list(range(1, 11))

with open("nosimd.txt") as f:
    nosimd = [float(line.strip()) for line in f]

with open("simd.txt") as f:
    simd = [float(line.strip()) for line in f]

# Debug check (important)
print(len(threads), len(nosimd), len(simd))

plt.plot(threads, nosimd, marker='o', label="No SIMD")
plt.plot(threads, simd, marker='o', label="SIMD")

plt.xlabel("Threads")
plt.ylabel("Time (ms)")
plt.title("Task2 Performance: SIMD vs No SIMD")
plt.legend()
plt.grid()

plt.savefig("task2.pdf")
