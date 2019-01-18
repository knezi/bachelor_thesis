#!/bin/env python3
import json
import datetime as dt
import sys
import subprocess as sp
import argparse
from subprocess import PIPE


def spell_check(text):
    """ takes the argument returns the parsed output of
    lang_recognition.sh"""

    p = sp.Popen(['./lang_recognition.sh'], stdout=PIPE, stdin=PIPE,
                 stderr=sys.stderr)
    stdout_data = p.communicate(input=text.encode('utf-8'))[0]
    res = [int(x) for x in stdout_data.decode('utf-8').strip().split('\n')]
    return res


def main(args):
    """ expects args to contain:
        from_date
        to_date
        lang_check"""

    min_date = dt.datetime(2100, 1, 1)
    max_date = dt.datetime(100, 1, 1)
    from_date = dt.datetime.fromisoformat(args.from_date)
    to_date = dt.datetime.fromisoformat(args.to_date)

    try:
        while True:
            d = json.loads(input().strip())
            date = dt.datetime(*[int(x) for x in d['date'].split("-")])
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

                print("{}".format(json.dumps(d)))

    except EOFError:
        pass


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="""Classical unix filter
                                        from stdin in json-line format filters
                                        out lines that are in a specified date
                                        range and sends them to stdou.
                                        Optionally non-german&non-french
                                        (utility lang_recognisition.sh) is
                                        used.""")

    argparser.add_argument("from_date", type=str,
                           help="in format YYYY-MM-HH")
    argparser.add_argument("to_date", type=str,
                           help="in format YYYY-MM-HH")
    argparser.add_argument("-l", "--lang-check", action='store_true',
                           help="non-english text will be filtered out ")

    main(argparser.parse_args(sys.argv[1:]))
