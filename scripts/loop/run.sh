#!/bin/zsh
# run.sh "<what>" <script.py args...>: run a python script from the repo root, appending a header and all output
# to out/training.log (the single log the user tails).
cd ${0:A:h}/../..
source .venv/bin/activate
: ${SF2_ROM:="$HOME/Downloads/sf2/Street Fighter II (USA).sfc"}; export SF2_ROM
what=$1; shift
mkdir -p out
echo "=== $(date +%H:%M:%S) loop: $what (python $*) ===" >> out/training.log
t0=$(date +%s)
python "$@" >> out/training.log 2>&1
rc=$?
echo "=== $(date +%H:%M:%S) loop: $what finished rc=$rc in $(( $(date +%s) - t0 ))s ===" >> out/training.log
exit $rc
