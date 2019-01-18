#!/bin/env python3
#TODO - popisky os
#TODO komentare atak popis formatu
import json
import subprocess as sp

def inc(dic, val):
    if val in dic:
        dic[val]+=1
    else:
        dic[val]=1

def get(dic, val):
    if val in dic:
        return dic[val]
    return 0

with open('data.json', 'r') as d, open('graph', 'w') as w:
    words=dict()
    err=dict()
    rate=[]
    for x in d:
        line=json.loads(x)
        inc(words, line['words'])
        inc(err, line['incorrect_words'])
        rate.append(line['incorrect_words']/line['words'])
    
        
    for x in range(min(words), max(words)+1):
        w.write("{}\n".format(get(words, x)))

    w.write("\n\n")

    for x in range(min(err), max(err)+1):
        w.write("{}\n".format(get(err, x)))

    w.write("\n\n")

    step=0.01
    x=0
    # range doesn't support float
    while x<=1:
        w.write("{} {}\n".format(x, len(list(filter(lambda y:y>=x and y<x+step, rate)))))
        x+=step

sp.call(['gnuplot', 'plot_graph'])
