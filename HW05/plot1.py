import numpy as np
import matplotlib.pyplot as plt

def read_times(fname):
    """Read only the time (ms) values for int, float, double from a file."""
    int_times = []
    float_times = []
    double_times = []
    with open(fname) as f:
        lines = f.readlines()

    # Each n has 9 lines: [C0_int, Cn_int, time_int, C0_float, Cn_float, time_float, C0_double, Cn_double, time_double]
    for i in range(0, len(lines), 9):
        int_times.append(float(lines[i+2].strip()))
        float_times.append(float(lines[i+5].strip()))
        double_times.append(float(lines[i+8].strip()))
    return int_times, float_times, double_times

# Read times for block_dim=16
int16, float16, double16 = read_times("times16.txt")
# Read times for block_dim=32
int32, float32, double32 = read_times("times32.txt")

# Generate n values dynamically based on number of entries
n16 = [2**i for i in range(5, 5 + len(int16))]
n32 = [2**i for i in range(5, 5 + len(int32))]

plt.figure(figsize=(10,6))

# Plot int
plt.plot(n16, int16, 'o-', label='int, block=16')
plt.plot(n32, int32, 'o--', label='int, block=32')

# Plot float
plt.plot(n16, float16, 's-', label='float, block=16')
plt.plot(n32, float32, 's--', label='float, block=32')

# Plot double
plt.plot(n16, double16, '^-', label='double, block=16')
plt.plot(n32, double32, '^--', label='double, block=32')

plt.xlabel('Matrix size n')
plt.ylabel('Time (ms)')
plt.title('Tiled Matrix Multiplication Performance for int, float, double')
plt.xscale('log', base=2)
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("task1_all_types.pdf")
