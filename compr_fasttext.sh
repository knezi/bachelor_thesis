#!/bin/sh
# TODO COMMENTS READING AND STUFF



f='compr'

#echo Total reviews:
total=`cat "$f" | wc -l `
#echo $total


#echo Correctly assesed:
corr=`grep "$f" -e '__label__useful __label__useful' -e '__label__not-useful __label__not-useful' | wc -l`
#echo $corr


#echo Accuracy:
python -c "print('accuracy', $corr/$total)"
