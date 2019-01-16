#!/bin/env python3
import json

try:
    while True:
        d=json.loads(input().strip())
        nd={}
        nd['stars']=d['stars']
        print(json.dumps(nd))

except EOFError as e:
    pass
