#!/bin/sh
# TODO CMENT
# $1 $2 $3
predicted="data/fasttext_predicted"
# train test_data test_lables
ft="fasttext/fastText-0.1.0/fasttext"

model='data/fasttext_model'
./$ft supervised -input "$1" -output "$model"
./$ft predict "$model.bin" "$2" >"$predicted"
paste -d' ' "$3" "$predicted" > 'compr'
./compr_fasttext.sh

#rm 'compr'
paste "$3" "$2" | ./$ft test "$model.bin" -
