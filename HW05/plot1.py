import numpy as np
import matplotlib.pyplot as plt

def read_all_types(fname):
    int_times = []
    float_times = []
    double_times = []
    with open(fname) as f:
        lines = f.readlines()
    # Each run outputs 3 lines: int, float, double
    for i in range(0, len(lines), 3):
        int_times.append(float(lines[i].strip()))
        float_times.append(float(lines[i+1].strip()))
        double_times.append(float(lines[i+2].strip()))
    return int_times, float_times, double_times

# n values: 2^5 ... 2^14
n = [2**i for i in range(5, 15)]

# Read times for block_dim=16
int16, float16, double16 = read_all_types("times16.txt")
# Read times for block_dim=32
int32, float32, double32 = read_all_types("times32.txt")

# Plotting
plt.figure(figsize=(10,6))

# int
plt.plot(n, int16, 'o-', label='int, block=16')
plt.plot(n, int32, 'o--', label='int, block=32')

# float
plt.plot(n, float16, 's-', label='float, block=16')
plt.plot(n, float32, 's--', label='float, block=32')

# double
plt.plot(n, double16, '^-', label='double, block=16')
plt.plot(n, double32, '^--', label='double, block=32')

plt.xlabel('Matrix size n')
plt.ylabel('Time (ms)')
plt.title('Tiled Matrix Multiplication Performance for int, float, double')
plt.xscale('log', base=2)
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("task1_all_types.pdf")
