#!/bin/env python3
import json
import datetime as dt
import sys
import subprocess as sp
from subprocess import PIPE

def spell_check(text):
	p=sp.Popen(['./spell_check.sh'], stdout=PIPE, stdin=PIPE, stderr=sys.stderr)
	stdout_data = p.communicate(input=text.encode('utf-8'))[0]
	res=[int(x) for x in stdout_data.decode('utf-8').strip().split('\n')]
	return res

min_date=dt.datetime(2100,1,1)
max_date=dt.datetime(100,1,1)
from_date=dt.datetime(2012, 5, 1)
to_date=dt.datetime(2012, 12, 1)
count=0
count_int=0

try:
	with open(sys.argv[1], 'w') as w:
		while True:
			if count_int%10000==0:
			    print(count_int)
			count_int+=1
			d=json.loads(input().strip())
			date=dt.datetime(*[int(x) for x in d['date'].split("-")])
			min_date=min(date, min_date)
			max_date=max(date, max_date)

			if from_date <= date <= to_date:
				# aspell considers dashes to be a comment
				text = d['text']
				ratio_en=spell_check(text)

				# text not in english
				if ratio_en[0]==-1:
					continue

				count+=1
				d['words']=ratio_en[1]
				d['incorrect_words']=ratio_en[0]

				w.write("{}\n".format(json.dumps(d)))


# except Exception as e:
    # print(e)
    # print(d)

except EOFError as e:
	pass


print(min_date)
print(max_date)
print("{} lines write from {} to {}.".format(count, from_date, to_date))
