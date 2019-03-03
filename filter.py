#!/bin/env python3
"""Classical unix filter. Take stdin in json-line format and write
to stdout lines that are in a specified date range.
Optionally german&french lines (utility lang_recognisition.sh) are filtered out.
"""

import json
import datetime as dt
import sys
import subprocess as sp
import argparse
from subprocess import PIPE


def spell_check(text):
    """Run lang_recognition.sh on the string given as argument.

    text - input string to the script
    Return list of script output lines converted to int (should be two lines)
    The script lang_recognition.sh outputs -1 if the string isn't English."""

    p = sp.Popen(['./lang_recognition.sh'], stdout=PIPE, stdin=PIPE,
                 stderr=sys.stderr)
    stdout_data = p.communicate(input=text.encode('utf-8'))[0]
    res = [int(x) for x in stdout_data.decode('utf-8').strip().split('\n')]
    return res


def filter(args):
    """Grep lines between the given dates and optionally in English from stdin.


    args must be namespace containing:
        from_date - format (YYYY-MM-HH) beginning of the span
        to_date - format (YYYY-MM-HH) end of the span
        lang_check - boolean if the language check should be done

    There must be data in stdin in JSON-line format (TODO link to spec)
    Writes to stdout only matching lines."""

    min_date = dt.datetime(2100, 1, 1)
    max_date = dt.datetime(100, 1, 1)
    from_date = dt.datetime.fromisoformat(args.from_date)
    to_date = dt.datetime.fromisoformat(args.to_date)

    try:
        while True:
            d = json.loads(input().strip())
            date = dt.datetime(*[int(x) for x in d['date'].split('-')])
            min_date = min(date, min_date)
            max_date = max(date, max_date)

            if from_date <= date <= to_date:
                # aspell considers dashes to be a comment
                text = d['text']
                ratio_en = spell_check(text)

                # text not in english
                if args.lang_check and ratio_en[0] == -1:
                    continue

                d['words'] = ratio_en[1]
                d['incorrect_words'] = ratio_en[0]

                print('{}'.format(json.dumps(d)))

    except EOFError:
        pass


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('from_date', type=str,
                           help='in format YYYY-MM-HH')
    argparser.add_argument('to_date', type=str,
                           help='in format YYYY-MM-HH')
    argparser.add_argument('-l', '--lang-check', action='store_true',
                           help='non-english text will be filtered out ')

    filter(argparser.parse_args(sys.argv[1:]))
