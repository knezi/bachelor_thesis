#!/bin/sh
echo $PWD
# Denormalises data from the original Yelp files such
# that each line of the output file contains review & business info

sort_wrapper() {
	# $1 - file
	# $2 - key position of sorting
	# $3 - out_file
	LC_ALL=C sort -t \" -k "$2" "$1" > "$3"
}


[ \( "$1" = "-h" \) -o \( "$#" -ne 3 \) ] &&
	printf "Usage: denormalise.sh in_dir out_file
in_dir		path to a directory with original Yelp files
out_file	output file\n
ids_file	file of sorted business ids (id per line)\n" &&
	exit 0

filter_exec='./denormalization/filter.py'
join_exec='./denormalization/join.py'
extract_ids_exec='./denormalization/extract_ids.py'


# sort businesses
business="$1/business_sorted.json"
sort_wrapper "$1/business.json" 4 "$business"


# filter only a particular time period and filter out german&french
tmp=`mktemp -p .`
tmp2=`mktemp -p .`
cat "$1/review.json" | PYTHONPATH=. $filter_exec -l 2012-05-01 2012-12-01 > "$tmp"

# join with businesses
sort_wrapper "$tmp" 10 "$tmp2"
mv "$tmp2" "$tmp"
PYTHONPATH=. $join_exec "$tmp" "$business" "$tmp2" "business_id"
mv "$tmp2" "$tmp"

# getting linguistics data - GENEEA
# extracting ids - used for subsequent data extraction
sort_wrapper "$tmp" 4 "$tmp2"
mv "$tmp2" "$tmp"
PYTHONPATH=. $extract_ids_exec "$tmp" "$3"

# the final data are sorted alphabetically with respect to review_id
# OUTPUT FILE
mv "$tmp" "$2"

# remove tmp files
rm -f "$business"
