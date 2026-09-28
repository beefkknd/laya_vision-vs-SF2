#!/bin/zsh
# gate.sh <model dir> <name>: the gate protocol in PROGRESS.md (20 paired matches vs Dhalsim).
S=${0:A:h}
$S/run.sh "gate $2 ($1)" scripts/parallel.py --workers 4 --base-port 47810 play_student --model $1 --name $2 \
  --matches 20 --seed 4242 --savestate states/chunli_vs_dhalsim.state --me chunli --opp dhalsim
