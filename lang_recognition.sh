#!/bin/sh
tmp=`mktemp`
cat > "$tmp"

en=`cat "$tmp" | aspell list -l "en_GB" --encoding utf-8 | wc -l`
de=`cat "$tmp" | aspell list -l "de_DE" --encoding utf-8 | wc -l`
fr=`cat "$tmp" | aspell list -l "fr" --encoding utf-8 | wc -l`

[ \( $en -le $de \) -a \( $en -le $fr \) ] && printf -- "en" && exit
[ \( $fr -le $de \) ] && printf -- "fr" && exit
printf -- "de" && exit
