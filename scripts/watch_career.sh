#!/bin/zsh
# ONE command: play a BLANK-start career (continuous, replay-on-loss, beat-and-move-on) AND watch it live.
#
#   ./scripts/watch_career.sh                 # chunli vs the 7-fighter ladder, TUI view (text gameplay)
#   ./scripts/watch_career.sh guile           # a different character
#   ./scripts/watch_career.sh chunli --watch  # ALSO open a visible SNES (Mesen) window for each match
#   ./scripts/watch_career.sh chunli --opps honda,guile --block 6   # any play_career.py flag passes through
#
# Ctrl-C stops BOTH the watch and the career. The Qwen server defaults to threebody over Tailscale;
# override by exporting SF2_QWEN_URL first.
set -u
cd "$(dirname "$0")/.."

ME="${1:-chunli}"
[ $# -gt 0 ] && shift                                   # the rest are split: --font -> monitor, else -> play_career
MON_ARGS=()
CAREER_ARGS=()
for a in "$@"; do
  case "$a" in
    --font) MON_ARGS+=(--font) ;;                       # Chinese title bars: a MONITOR arg, not a career one
    *) CAREER_ARGS+=("$a") ;;
  esac
done
export SF2_QWEN_URL="${SF2_QWEN_URL:-http://100.66.12.33:8080/v1/chat/completions}"
PY=.venv/bin/python
NAME="career_${ME}_$(date +%s)"
SESS="rollouts/career/${NAME}"
LOG="/tmp/${NAME}.log"

echo "worktree  : $(pwd)"
echo "branch    : $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
echo "character : ${ME}"
echo "qwen      : ${SF2_QWEN_URL}"
echo "session   : ${SESS}"
echo "driver log: ${LOG}"
echo

# start the continuous career in the background; its stdout/stderr go to the log
$PY scripts/play_career.py --me "${ME}" --name "${NAME}" "${CAREER_ARGS[@]}" > "${LOG}" 2>&1 &
CAREER_PID=$!
cleanup() { kill ${CAREER_PID} 2>/dev/null; }
trap cleanup EXIT INT TERM

# wait for the driver to pass its Qwen preflight and create the session dir (or die early)
for _ in {1..120}; do
  [ -d "${SESS}" ] && break
  if ! kill -0 ${CAREER_PID} 2>/dev/null; then
    echo "career driver exited before it started playing -- see below:"; echo; cat "${LOG}"; exit 1
  fi
  sleep 0.5
done

echo "watching live (Ctrl-C to stop both)..."
sleep 1
# the TUI tails the session: per-opponent rounds + CAREER TREND + live gameplay + DATA FLOW pulse
$PY scripts/monitor_tui.py --session "${SESS}" --no-grade "${MON_ARGS[@]}"
