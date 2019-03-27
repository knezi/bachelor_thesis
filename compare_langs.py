#!/bin/env python3
import argparse
import json
import sys
import itertools
from geneea.analyzer.model import f2converter
from geneea.analyzer.model.x3 import X3
from subprocess import PIPE
import subprocess as sp

def spell_check(text):
    p = sp.Popen(['./spell_check.sh'], stdout=PIPE, stdin=PIPE, stderr=sys.stderr)
    stdout_data = p.communicate(input=text.encode('utf-8'))[0]
    res = [x for x in stdout_data.decode('utf-8').strip().split('\n')]
    return res[0]




def main(args):
    global analysis
    with open(args.analysisIn, encoding='utf8') if args.analysisIn \
            else sys.stdin as analysisReader:
        if args.max >= 0:
            analysisReader = itertools.islice(analysisReader, 0, args.max)

        total = 0
        langs = dict()
        allowed = ['en', 'de', 'fr']
        for raw in map(json.loads, analysisReader):
            total += 1
            analysis = f2converter.fromDict(raw)
            lang = spell_check(raw['text'])
            # print(lang, analysis.language.detected, analysis.docId)
            if analysis.language.detected not in allowed:
                continue
            if lang != analysis.language.detected:
                print(lang, analysis.language.detected, analysis.docId, raw['text'])
            
            if not lang in langs:
                langs[lang] = 1
            else:
                langs[lang] += 1

            if total % 1000 == 0:
                print(f'dump {total}')

        print(f'Total processed {total}.')
        print(langs)




if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--analysisIn',
                           help='Result of Full Analysis, one json per line')
    argparser.add_argument('-n', '--max', type=int,
                           help='Maximal number of processed analysis documents', default=-1)
    main(argparser.parse_args(sys.argv[1:]))
