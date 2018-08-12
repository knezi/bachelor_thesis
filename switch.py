#!/bin/env python3
import json

with open('data.json', 'r') as d, open('d', 'w') as w:
    dots=[]
    for x in d:
        line=json.loads(x)
        line['words'],line['incorrect_words']=line['incorrect_words'],line['words']
        w.write("{}\n".format(json.dumps(line)))

