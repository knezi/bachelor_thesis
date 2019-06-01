#!/bin/sh
# many files have their unittest run if you run them as a main file
# plus my_unittest.py is a big unittest suite

function runtest() {
	PYTHONPATH=. "$1"
	ret="$?"
	if [ "$ret" != 0 ]; then
		exit "$ret"
	fi
}

for x in ./my_unittests.py ./statistics.py ./utils.py ./preprocessors/featurematrixconversion.py ./preprocessors/featureselectionbase.py ./preprocessors/chisquare.py ./preprocessors/mutualinformation.py;
do
	runtest $x
done
