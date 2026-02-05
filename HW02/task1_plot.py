import matplotlib.pyplot as plt

n_values = [2**i for i in range(10, 31)]
times = []

with open("task1_times.txt") as f:
    for line in f:
        line = line.strip()
        if line:
            times.append(float(line))

assert len(times) == len(n_values), f"Mismatch: {len(times)} vs {len(n_values)}"

plt.figure(figsize=(8,6))
plt.plot(n_values, times, marker='o', linestyle='-')
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel("Array size n")
plt.ylabel("Scan time (ms)")
plt.title("Scaling analysis of inclusive scan")
plt.grid(True, which="both", ls="--")
plt.savefig("task1.pdf")
