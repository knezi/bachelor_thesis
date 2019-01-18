#!/bin/sh
if [ "$1" = "-h" ]; then
	cat <<EOF
language recognistion tool for English, German and French.
The given text on stdin processed with a spell checker and chooses the language
with the lowest numbers of incorrect words

If option '-l' is specified, writes detected language to the output
  ('en', 'fr' or 'de')

Otherwise returns two line separated numbers:
for non-english documents -1\n0
for english number of incorrect words\n number of all words
EOF
	exit
fi

function exit_properly() {
    rm "$tmp"
    exit
}

tmp=`mktemp`
cat > "$tmp"

# get number of incorrect words
en=`cat "$tmp" | aspell list -l "en_GB" --encoding utf-8 | wc -l`
de=`cat "$tmp" | aspell list -l "de_DE" --encoding utf-8 | wc -l`
fr=`cat "$tmp" | aspell list -l "fr" --encoding utf-8 | wc -l`


if [ "$1" = "-l" ]; then
    # list language
    [ \( $en -le $de \) -a \( $en -le $fr \) ] && printf -- "en" && exit_properly
    [ \( $fr -le $de \) ] && printf -- "fr" && exit_properly
    printf -- "de" && exit_properly
else

    # get spell checks rates
    if [ \( $en -gt $de \) -o \( $en -gt $fr \) ];
    then
        printf -- "-1\n0"
    else
        printf "$en\n"
        cat "$tmp" | wc -w
    fi
fi

exit_properly
