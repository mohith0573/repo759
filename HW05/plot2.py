import numpy as np
import matplotlib.pyplot as plt

def read(fname):
    vals=[]
    with open(fname) as f:
        lines=f.readlines()
    for i in range(1,len(lines),2):
        vals.append(float(lines[i]))
    return vals

N=[2**i for i in range(10,21)]
t1=read("times1024.txt")
t2=read("times256.txt")

plt.plot(N,t1,label="tpb=1024")
plt.plot(N,t2,label="tpb=256")
plt.legend()
plt.xlabel("N")
plt.ylabel("ms")
plt.savefig("task2.pdf")
