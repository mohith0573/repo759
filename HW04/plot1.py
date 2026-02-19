import numpy as np
import matplotlib.pyplot as plt

def load(fname):
    with open(fname) as f:
        return [float(x.strip()) for x in f if x.strip()]

n1 = [2**i for i in range(5,15)]
t1_1024 = load("task1_times_1024.txt")
t1_256  = load("task1_times_256.txt")

plt.figure()
plt.plot(n1,t1_1024,label="tpb=1024")
plt.plot(n1,t1_256,label="tpb=256")
plt.xscale("log",base=2)
plt.xlabel("n")
plt.ylabel("Time (ms)")
plt.legend()
plt.savefig("task1.pdf")

