#!/bin/env python3

# TODO argparse
import json
import sys

ids = sys.argv[1]
lings = sys.argv[2]
out = sys.argv[3]

with open(ids, 'r') as i, open(lings, 'r') as ling, open(out, 'w') as w:
    ids = set()
    for x in i:
        ids.add(x.strip())

    for line in ling:
        analysis = json.loads(line)
        if analysis['id'] in ids:
            w.write(f'{line}\n')
