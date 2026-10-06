#!/bin/zsh
# ONE command: watch Chun-Li play at two skill levels -- a visible Mesen (SNES) window AND the live TUI.
#
#   ./scripts/watch_chunli.sh early      # EARLY  (thin) table  ->  she wins ~30% (loses most)
#   ./scripts/watch_chunli.sh expert     # EXPERT (trained) table ->  she wins most (~67% rounds, every match)
#   ./scripts/watch_chunli.sh expert --opp ken --speed 100   # extra play_loop_screen args pass through
#
# Full System-1 quorum decides every move (laya + 3 bees + the value table). The run is FROZEN
# (--no-learn --explore 0), so the skill level stays STABLE for the demo and NO Qwen server is needed.
# The only thing that changes between the two modes is the value-table file -- that IS the skill.
# Ctrl-C stops BOTH the Mesen window and the TUI.
set -u
cd "$(dirname "$0")/.."

MODE="${1:-expert}"; [ $# -gt 0 ] && shift
export SF2_ROM="${SF2_ROM:-$PWD/roms/Street Fighter II (USA).sfc}"
PY=.venv/bin/python
OPP="ryu"

case "$MODE" in
  early)  TABLE="runs/tables/chunli_early.json"; DESC="EARLY  table  ->  wins ~30% (loses most)"
          # the early (thin) table is derived from the trained one; make it on first use
          [ -f "$TABLE" ] || $PY scripts/make_early_table.py runs/tables/chunli.json "$TABLE";;
  expert) TABLE="runs/tables/chunli.json";       DESC="EXPERT table  ->  wins most (~67% rounds, every match)";;
  -h|--help) echo "usage: $(basename "$0") {early|expert} [extra play_loop_screen args]"; exit 0;;
  *) echo "unknown mode: '$MODE'  (use 'early' or 'expert')"; exit 1;;
esac
[ -f "$TABLE" ] || { echo "missing table file: $TABLE"; exit 1; }

OUT="rollouts/demo/chunli_${MODE}_$(date +%s)"
echo "mode  : $MODE  --  $DESC"
echo "table : $TABLE"
echo "opp   : $OPP      out: $OUT"
echo "(Ctrl-C stops both the Mesen window and the TUI)"
echo

# System 1 quorum, frozen: the carried table sets the skill. Visible Mesen window; Script Window parked off-screen.
$PY scripts/play_loop_screen.py --me chunli --opp "$OPP" --policy quorum --quorum-mode vote \
  --cat-advisor runs/text_laya/cat_v4 --move-advisor runs/text_laya/move_v3 \
  --carry-table "$TABLE" --no-learn --explore 0 \
  --watch --speed 150 --hide-console --games 20 --rounds 3 --out "$OUT" "$@" &
PLAY=$!
trap 'kill $PLAY 2>/dev/null' EXIT INT TERM

# the TUI refuses to tail an empty dir -- wait for the first round's gameplay, then attach it
for _ in {1..240}; do
  ls "$OUT"/g*_r*/decisions.jsonl >/dev/null 2>&1 && break
  kill -0 $PLAY 2>/dev/null || { echo; echo "game process exited before any gameplay -- see output above."; wait $PLAY; exit 1; }
  sleep 0.5
done
sleep 0.5
$PY scripts/monitor_tui.py --watch "$OUT" --no-grade
