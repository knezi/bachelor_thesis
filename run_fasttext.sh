#!/bin/sh
# TODO CMENT
# TODO silent -q ?
# todo compr
# $1 - path prefix
train="$1_train"
test_data="$1_test_data"
test_lables="$1_test_lables"
predicted="$1_predicted"
model="$1_model"

ft="./fasttext/fastText-0.1.0/fasttext"

$ft supervised -verbose 0 -input "$train" -output "$model" -epoch 50 -wordNgrams 3
$ft predict "$model.bin" "$test_data" >"$predicted"
paste -d' ' "$test_lables" "$predicted" > 'compr'

./compr_fasttext.sh

rm 'compr'
paste "$test_lables" "$test_data" | ./$ft test "$model.bin" - 2>/dev/null| grep "@" \
	| sed "s/P@1	/precision /;s/R@1	/recall /"
