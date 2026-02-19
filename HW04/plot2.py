import numpy as np
import matplotlib.pyplot as plt

def load(fname):
    with open(fname) as f:
        return [float(x.strip()) for x in f if x.strip()]

n2 = [2**i for i in range(10,30)]
t2_1024 = load("task2_times_1024.txt")
t2_256  = load("task2_times_256.txt")

plt.figure()
plt.plot(n2,t2_1024,label="tpb=1024")
plt.plot(n2,t2_256,label="tpb=256")
plt.xscale("log",base=2)
plt.xlabel("n")
plt.ylabel("Time (ms)")
plt.legend()
plt.savefig("task2.pdf")
