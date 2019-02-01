#!/bin/env python3
# joins two given files based on the key given in the fourth argument
import json
import sys
import argparse


def join(args):
    """ expects args to contain: file1, file2, outfile, key"""

    with open(args.file1, 'r') as file1, \
            open(args.file2, 'r') as file2, \
            open(args.outfile, 'w') as outfile:
        cursor = json.loads(file2.readline())

        for x in file1:
            line = json.loads(x)
            while cursor[args.key] < line[args.key]:
                cursor = json.loads(file2.readline())

            line[args.key] = cursor

            outfile.write('{}\n'.format(json.dumps(line)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="""Joins two given files
                                        based on the key given in the fourth
                                        argument. The occurence of the key in
                                        file1 is replaced with a json-line of
                                        file2.""")

    argparser.add_argument('file1', type=str,
                           help='file1 (whose key will be replaced)')
    argparser.add_argument('file2', type=str,
                           help='file2 (which will replace the key)')
    argparser.add_argument('outfile', type=str,
                           help='the resulting file')
    argparser.add_argument('key', type=str,
                           help='key of json on which the join is made')

    join(argparser.parse_args(sys.argv[1:]))
