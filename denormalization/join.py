#!/bin/env python3
"""Line-wise join two files in JSON-line by replacing values of the given key
in the lines of the first file by the corresponding lines from the second file.
"""
import json
import sys
import argparse


def join(args):
    """Join the files being specified in the argument  as defined above.

    :param args: args must be namespace containing:
        file1 - path to the file whose key are being replaced
        file2 - path to the  file which will replaced the keys
        outfile - where the resulting file will be created
        key - name of the JSON field in the first file
    """

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
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('file1', type=str,
                           help='file1 (whose key will be replaced)')
    argparser.add_argument('file2', type=str,
                           help='file2 (whose lines will replace the key)')
    argparser.add_argument('outfile', type=str,
                           help='the resulting file')
    argparser.add_argument('key', type=str,
                           help='key of json on which the join is made')

    join(argparser.parse_args(sys.argv[1:]))
