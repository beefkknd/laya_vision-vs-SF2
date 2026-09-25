#!/usr/bin/env bash
# One turn of the loop: student plays -> DAgger relabel -> LoRA on seed + new rows -> gate.
#   scripts/dagger_round.sh <round number>      e.g. scripts/dagger_round.sh 1   (needs runs/r0/best)
# Environment:
#   WORKERS=4      play on 4 headless Mesen workers in parallel (needs SF2_ROM; SF2_MESEN if not in /Applications)
#                  WORKERS=1 (default) uses the Mesen window you have open with the bridge script loaded
#   MATCHES=12     student matches per round (split across workers)
#   SAVESTATE=...  fight-start savestate         EXTRA="--me chunli --opp guile"  passed to every play run
set -euo pipefail
N=${1:?round number}; P=$((N - 1)); MATCHES=${MATCHES:-12}; WORKERS=${WORKERS:-1}
SAVESTATE=${SAVESTATE:-states/ryu_vs_ken.state}; read -r -a EXTRA <<< "${EXTRA:-}"
cd "$(dirname "$0")/.."

play() {  # play <model> <name>
  if [ "$WORKERS" -gt 1 ]; then
    python scripts/parallel.py --workers "$WORKERS" play_student --model "$1" --name "$2" --matches "$MATCHES" \
      --savestate "$SAVESTATE" ${EXTRA[@]+"${EXTRA[@]}"}
  else
    python scripts/play_student.py --model "$1" --name "$2" --matches "$MATCHES" --savestate "$SAVESTATE" ${EXTRA[@]+"${EXTRA[@]}"}
  fi
}

[ -f rollouts/r$P/gate.json ] || play runs/r$P/best r$P
python scripts/relabel.py --rollout rollouts/r$P --name dagger_r$N
DATA=(--data data/seed_teacher)
for i in $(seq 1 "$N"); do
  DATA+=(--data data/dagger_r$i)
  [ -d data/dagger_r${i}_hot ] && DATA+=(--data data/dagger_r${i}_hot)
done
python scripts/train.py "${DATA[@]}" --init runs/r$P/best --out runs/r$N --epochs 1
play runs/r$N/best r$N
python scripts/gate.py rollouts/teacher rollouts/r$P rollouts/r$N
