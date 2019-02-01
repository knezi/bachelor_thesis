#!/bin/env python3

# TODO argparse
import json
import sys

file = sys.argv[1]
out = sys.argv[2]

with open(file, 'r') as r, open(out, 'w') as w:
    for l in r:
        i = json.loads(l)['review_id']
        w.write('{}\n'.format(i))
