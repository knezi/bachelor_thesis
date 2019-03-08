#!/bin/sh
[ \( "$1" = "-h" \) -o \( "$#" -ne 1 \) ] &&
    printf "Usage: compr_fasttext.sh compr_file

Computes accuracy of fasttext model based on predicted and actual labels.

compr_file      each line contains the actual label and predicted
                separated by a space.\n" && exit 0

#Total reviews:
total=`cat "$1" | wc -l `

#Correctly assesed:
corr=`grep "$1" -e '__label__useful __label__useful' -e '__label__not-useful __label__not-useful' | wc -l`


#Accuracy:
python -c "print('accuracy', $corr/$total)"
