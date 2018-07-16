#!/bin/sh

sort_here() {
	# $1 - file
	# $2 - key position of sorting
	# $3 - out_file
	LC_ALL=C sort -t \" -k "$2" "$1" > "$3"
}

[ \( "$1" = "-h" \) -o \( "$#" -ne 2 \) ] &&
	printf "Usage: create_data.sh in_dir out_file
in_dir		path to a directory with original Yelp files
out_file	output file\n" &&
	exit 1


# sort businesses & users
business="$1/business_red.json"
user="$1/user_red.json"

sort_here "$1/business.json" 4 "$business"
sort_here "$1/user.json" 4 "$user"


# tmp=`mktemp -p .`
# cat "$user" | ./reduce_user.py > "$tmp"
# mv "$tmp" "$user"


# filter only a particular time period
tmp=`mktemp -p .`
tmp2=`mktemp -p .`
cat "$1/review.json" | ./filter.py "$tmp"

# join with businesses
echo joining businesses
sort_here "$tmp" 10 "$tmp2"
mv "$tmp2" "$tmp"
./join.py "$tmp" "$business" "$tmp2" "business_id"
mv "$tmp2" "$tmp"

# join with users
# echo joining users
# sort_inplace "$tmp" 8
# ./join.py "$tmp" "$user" "$tmp2" "user_id"
# mv "$tmp2" "$tmp"


# OUTPUT FILE
mv "$tmp" "$2"
