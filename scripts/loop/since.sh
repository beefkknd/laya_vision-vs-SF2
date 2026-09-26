#!/bin/zsh
# since.sh "<header text>": print out/training.log from the first header line containing it.
cd ${0:A:h}/../..
n=$(grep -n -F "$1" out/training.log | head -1 | cut -d: -f1)
tail -n +${n:-1} out/training.log
