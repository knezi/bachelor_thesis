#!/bin/env python3
# TODO AUTHOR
"""Reduce geenea files by removing all unused data.

It takes geenea file, extracts from it only used data
(sentiment, entities and id) and saves to another file."""
import sys
import json
from geneea.analyzer.model import f2converter
import argparse


def main(args: argparse.ArgumentParser) -> None:
    """Reduce geneea files by removing unused data."""
    with open(args.in_file, 'r', encoding='utf8') as analysisReader, \
            open(args.out_file, 'w', encoding='utf-8') as w:

        # each line is parsed as json and then
        # loaded with f2converter
        for raw in map(json.loads, analysisReader):
            analysis = f2converter.fromDict(raw)

            # new cropped line of a file
            line = dict()
            line['sentiment'] = analysis.sentiment.label \
                if analysis.sentiment is not None else 'n/a'
            line['id'] = analysis.docId
            # we have to convert entities from and to dict to avoid
            # parsing errors and json encoding
            line['entities'] = f2converter.toDict(analysis)['entities']

            w.write(f'{json.dumps(line)}\n')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('in_file', type=str,
                           help='File with geneea data')
    argparser.add_argument('out_file', type=str,
                           help='Resulting file')

    main(argparser.parse_args(sys.argv[1:]))
