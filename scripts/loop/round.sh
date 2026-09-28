#!/bin/zsh
# round.sh N PREV: one DAgger round. Relabel rollouts/<PREV>_gate -> data/dagger5_rN, train runs/rN from
# runs/<PREV>/best on the seed set plus every dagger set so far (hot sets at half weight), then gate it as rN_gate.
#   scripts/loop/round.sh 3 r2
S=${0:A:h}; N=$1; P=$2
cd $S/../..
[ -d data/dagger5_r$N ] || $S/run.sh "relabel ${P}_gate -> dagger5_r$N" scripts/relabel.py \
  --rollout rollouts/${P}_gate --name dagger5_r$N --val-every 5 || exit 1
DATA=(--data data/seed5_g10); MIX=()
for i in $(seq 1 $N); do
  DATA+=(--data data/dagger5_r$i)
  [ -d data/dagger5_r${i}_hot ] && { DATA+=(--data data/dagger5_r${i}_hot); MIX+=(--mix dagger5_r${i}_hot=0.5); }
done
$S/run.sh "train r$N (init $P, seed + dagger rounds 1..$N)" scripts/train.py $DATA $MIX --val-data data/eval5 \
  --init runs/$P/best --out runs/r$N --epochs 1 --eval-every 1000 --patience 3 --val-limit 8000 || exit 1
$S/gate.sh runs/r$N/best r${N}_gate
