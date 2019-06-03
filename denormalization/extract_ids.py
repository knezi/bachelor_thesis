#!/bin/env python3
"""Take a JSON-line file line-by-line and write out values of 'review_id'."""

import argparse
import json
import sys


def extract_ids(args: argparse.ArgumentParser) -> None:
    """Take a JSON-line file and write out values of the key 'review_id'.

    :param args: namespace which must contain:
                 from_file - JSON-line file being read
                 to_file - output file
    """
    with open(args.from_file, 'r') as r, open(args.to_file, 'w') as w:
        for l in r:
            i = json.loads(l)['review_id']
            w.write(f'{i}\n')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('from_file', type=str,
                           help='JSON-line file being read')
    argparser.add_argument('to_file', type=str,
                           help='output file')

    extract_ids(argparser.parse_args(sys.argv[1:]))
