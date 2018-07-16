#!/bin/env python3
# joins two given file based on the key given in fourth argument
# IN ARGUMENTS:
# reviews users out_file key_position
import json
import sys

key=sys.argv[4]

with open(sys.argv[1], 'r') as rev,\
	open(sys.argv[2], 'r') as usr,\
	open(sys.argv[3], 'w') as out:
	last=''
	us=json.loads(usr.readline())

	for x in rev:
		l=json.loads(x)
		while us[key]<l[key]:
			us=json.loads(usr.readline())

		l[key]=us

		out.write('{}\n'.format(json.dumps(l)))
