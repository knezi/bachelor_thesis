#!/bin/sh
tmp=`mktemp`
cat > "$tmp"

en=`cat "$tmp" | aspell list -l "en_GB" --encoding utf-8 | wc -l`
de=`cat "$tmp" | aspell list -l "de_DE" --encoding utf-8 | wc -l`
fr=`cat "$tmp" | aspell list -l "fr" --encoding utf-8 | wc -l`

[ \( $en -gt $de \) -o \( $en -gt $fr \) ] && printf -- "-1\n0" && exit


printf "$en\n"
cat "$tmp" | wc -w

# cat "$tmp" | aspell pipe -l "$1" --encoding utf-8 | head -n -1 | tail -n +2 | awk "
# BEGIN{total=0; inc=0};
# // {total +=1};
# /&/ {inc +=1};
# END{print total \" \" inc}
# " -
