#!/bin/env python3
# TODO AUTHOR
# extracts sentiment from geneea analysis

import sys
import json
from geneea.analyzer.model import f2converter
from geneea.analyzer.model.x3 import X3


with open("data/geneea_data.json", encoding='utf8') as analysisReader, \
     open("data/geneea_data_extracted.json", 'w', encoding='utf-8') as w:

    for raw in map(json.loads, analysisReader):
        analysis = f2converter.fromDict(raw)
        line = dict()
        line['sentiment'] = analysis.sentiment.label \
            if analysis.sentiment is not None else 'n/a'
        line['id'] = analysis.docId

        w.write("{}\n".format(json.dumps(line)))


# TODO ARGUMENTS FOLLOW CONVENTIONS


