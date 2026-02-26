import numpy as np
import matplotlib.pyplot as plt

def read(fname):
    vals=[]
    with open(fname) as f:
        lines=f.readlines()
    for i in range(2,len(lines),3):
        vals.append(float(lines[i]))
    return vals

n=[2**i for i in range(5,15)]
t16=read("times16.txt")
t32=read("times32.txt")

plt.plot(n,t16,label="block=16")
plt.plot(n,t32,label="block=32")
plt.legend()
plt.xlabel("n")
plt.ylabel("ms")
plt.savefig("task1.pdf")
