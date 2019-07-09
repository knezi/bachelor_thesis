#!/bin/env python3
import sys

with open(sys.argv[1], 'r') as r:
    labels = r.readline().split(';')
    data = list(map(float, r.readline().strip().split(';')))

    while len(data) != 0:
        print(labels[0])
        print(f'{data[1]:.2} ({data[2]:.2}, {data[3]:.2}) $\pm$ {data[4]:.2}')

        labels = labels[5:]
        data = data[5:]


