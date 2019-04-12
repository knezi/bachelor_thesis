#!/bin/env python3
# TODO AUTHOR
"""Reduce geenea files by removing all unused data.

It takes geenea file, extracts from it only used data
(sentiment, entities and id) and saves to another file."""
import sys
import json
import argparse


def main(args: argparse.ArgumentParser) -> None:
    """Reduce geneea files by removing unused data."""
    with open(args.in_file, 'r', encoding='utf8') as analysisReader, \
            open(args.out_file, 'w', encoding='utf-8') as w:

        used_features = ('sentiment', 'id', 'entities')
        # each line is parsed as json and then cropped
        # there's no need to load it with geneea analyzer
        for raw in map(json.loads, analysisReader):
            # new cropped line
            line = dict()
            for f in used_features:
                if f in raw:
                    line[f] = raw[f]

            w.write(f'{json.dumps(line)}\n')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('in_file', type=str,
                           help='File with geneea data')
    argparser.add_argument('out_file', type=str,
                           help='Resulting file')

    main(argparser.parse_args(sys.argv[1:]))
