#!/bin/sh
cat | aspell pipe -l en_GB | head -n -1 | tail -n +2 | awk "
BEGIN{total=0; inc=0};
// {total +=1};
/&/ {inc +=1};
END{print total \" \" inc}
" -
