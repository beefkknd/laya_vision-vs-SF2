#!/usr/bin/env bash
# One turn of the loop: student plays -> DAgger relabel -> LoRA on seed + new rows -> gate.
#   scripts/dagger_round.sh <round number>      e.g. scripts/dagger_round.sh 1   (needs runs/r0/best)
set -euo pipefail
N=${1:?round number}; P=$((N - 1)); MATCHES=${MATCHES:-10}; STATE=${STATE:-Champion.Level1.RyuVsGuile}
cd "$(dirname "$0")/.."

[ -f rollouts/r$P/gate.json ] || \
  python scripts/play_student.py --model runs/r$P/best --name r$P --matches "$MATCHES" --state "$STATE"
python scripts/relabel.py --rollout rollouts/r$P --name dagger_r$N
DATA=(--data data/seed_teacher)
for i in $(seq 1 "$N"); do
  DATA+=(--data data/dagger_r$i)
  [ -d data/dagger_r${i}_hot ] && DATA+=(--data data/dagger_r${i}_hot)
done
python scripts/train.py "${DATA[@]}" --init runs/r$P/best --out runs/r$N --epochs 1
python scripts/play_student.py --model runs/r$N/best --name r$N --matches "$MATCHES" --state "$STATE"
python scripts/gate.py rollouts/teacher rollouts/r$P rollouts/r$N
