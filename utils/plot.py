#!/bin/env python3
import sys
from matplotlib import pyplot as pp
import numpy as np

with open(sys.argv[1], 'r') as r:
    data = [x.strip().split('\t') for x in r]

x, y = tuple(zip(*data))

print(len(x))
print(len(y))

x = list(map(int, x))
y = list(map(float, y))
# x = x[::10]
# y = y[::10]
pp.plot(x, y)
pp.xscale('log')
# pp.xticks(np.arange(0, max(map(int, x)), step=10000))
pp.savefig('out.png', dpi=300)
