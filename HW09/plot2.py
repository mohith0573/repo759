import matplotlib.pyplot as plt

threads = list(range(1,11))

def read(fname):
    times = []
    with open(fname) as f:
        lines = f.readlines()
        for i in range(1, len(lines), 2):
            times.append(float(lines[i]))
    return times

nosimd = read("nosimd.txt")
simd = read("simd.txt")

plt.plot(threads, nosimd, label="No SIMD")
plt.plot(threads, simd, label="SIMD")

plt.xlabel("Threads")
plt.ylabel("Time (ms)")
plt.legend()
plt.grid()
plt.savefig("task2.pdf")
