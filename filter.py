#!/bin/env python3
"""Classical unix filter. Take stdin in json-line format and write
to stdout lines that are in a specified date range.
Optionally german&french lines (utility lang_recognisition.sh) are filtered out.
"""
from datetime import datetime

from typing import Tuple, List

import json
import datetime as dt
import sys
import subprocess as sp
import argparse
from subprocess import PIPE


def spell_check(text: str) -> Tuple[int, int]:
    """Run spell_check.sh on the string given as argument.

    spell_check.sh by default returns two-element array
    [# incorrect words, # all words] for English documents.
    [-1, 0] otherwise.
    Elements are separated by \n.

    :param text: input string to the script
    :return: tuple of exactly two numbers returned by spell_check.sh"""
    p:sp.Popen = sp.Popen(['./spell_check.sh'], stdout=PIPE, stdin=PIPE,
                          stderr=sys.stderr)

    stdout_data:List[str] = p.communicate(input=text.encode('utf-8'))[0]\
        .decode('utf-8')\
        .strip()\
        .split('\n')

    return (int(stdout_data[0]), int(stdout_data[1]))


def filter(args):
    """Grep lines between the given dates and optionally in English from stdin.

    :param args:
        args must be namespace containing:
            from_date - format (YYYY-MM-DD) beginning of the span
            to_date - format (YYYY-MM-DD) end of the span
            lang_check - boolean if the language check should be done

    Expects data in stdin in JSON-line format (TODO link to spec).
    Write only matching lines to stdout.
    """
    from_date = dt.datetime.fromisoformat(args.from_date)
    to_date = dt.datetime.fromisoformat(args.to_date)

    for line in map(lambda x: json.loads(x.strip()), sys.stdin):
        date: datetime = dt.datetime.fromisoformat(line['date'])

        if from_date <= date <= to_date:
            # aspell considers dashes to be a comment TODO? Wh?
            text = line['text']
            incorrect_words, words = spell_check(text)

            # text not in english
            if args.lang_check and incorrect_words == -1:
                continue

            line['words'] = words
            line['incorrect_words'] = incorrect_words

            print(json.dumps(line))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('from_date', type=str,
                           help='in format YYYY-MM-DD')
    argparser.add_argument('to_date', type=str,
                           help='in format YYYY-MM-DD')
    argparser.add_argument('-l', '--lang-check', action='store_true',
                           help='non-english text will be filtered out ')

    filter(argparser.parse_args(sys.argv[1:]))
