#!/bin/env python3
import json
import datetime as dt
import sys


min_date=dt.datetime(2100,1,1)
max_date=dt.datetime(100,1,1)
from_date=dt.datetime(2012, 5, 1) # first half of may
to_date=dt.datetime(2012, 8, 1)
count=0

try:
    with open(sys.argv[1], 'w') as w:
        while True:
            d=json.loads(input().strip())
            date=dt.datetime(*[int(x) for x in d['date'].split("-")])
            min_date=min(date, min_date)
            max_date=max(date, max_date)
            
            if from_date <= date <= to_date:
                count+=1
                w.write("{}\n".format(d))


except EOFError as e:
    pass

print(min_date)
print(max_date)
print("{} lines write from {} to {}.".format(count, from_date, to_date))
