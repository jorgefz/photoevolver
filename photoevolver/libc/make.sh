#!/bin/bash
temp=$( realpath "$0"  ) && dir=$(dirname "$temp")

echo $dir/shared/libc.so

gcc -Wall -Wextra -O2 -shared -o $dir/shared/libc.so -fPIC $dir/src/mloss.c $dir/src/struct.c


