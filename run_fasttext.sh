#!/bin/sh
[ \( "$1" = "-h" \) -o \( "$#" -lt 1 \) ] &&
    printf "Usage: run_fasttext.sh path_prefix [-v]
Runs fasttext on given data and computes various metrics about the generated model.

path_prefix     expected files are $path_prefix_{train,test_data,test_lables}
                train contains training data in format __label__... text+features
                test_data the same, but without __label__...
                test_lables contains only __label__... (the order must fit to test_data)

-v              for verbose output of fasttext\n" && exit 0


if [ "$1" = "-v" ]; then
	verbosity=1
	shift 1
else
	verbosity=0
	[ "$2" = "-v" ] && verbosity=1
fi

# TODO silent -q ?
# todo compr
# $1 - path prefix
train="$1_train"
test_data="$1_test_data"
test_lables="$1_test_lables"
predicted="$1_predicted"
model="$1_model"

ft="./fastText/fasttext"

$ft supervised -verbose "$verbosity" -input "$train" -output "$model" -epoch 50 -wordNgrams 3
$ft predict "$model.bin" "$test_data" >"$predicted"

# prints accuracy
paste -d' ' "$test_lables" "$predicted" > 'compr'
./compr_fasttext.sh 'compr'
rm 'compr'

# prints precision and recall
paste "$test_lables" "$test_data" | ./$ft test "$model.bin" - 2>/dev/null| grep "@" \
	| sed "s/P@1	/precision /;s/R@1	/recall /"
