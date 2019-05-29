#!/bin/sh
# many files have their unittest run if you run them as a main file
# plus my_unittest.py is a big unittest suite
PYTHONPATH=. ./my_unittests.py
PYTHONPATH=. ./statistics.py
PYTHONPATH=. ./utils.py
PYTHONPATH=. ./preprocessors/featurematrixconversion.py
PYTHONPATH=. ./preprocessors/featureselectionbase.py
PYTHONPATH=. ./preprocessors/chisquare.py
PYTHONPATH=. ./preprocessors/mutualinformation.py
