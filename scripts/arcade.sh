#!/bin/zsh
# Open the arcade console in its own Terminal window: laya plays Chun-Li in N Mesen windows, forever.
#   scripts/arcade.sh                      # 3 games, runs/r1/best
#   scripts/arcade.sh 2 runs/r2/best       # games, model
# Ctrl-C in the console (or closing it) stops every game. Drag the windows wherever you like.
cd "$(dirname "$0")/.." || exit 1
GAMES=${1:-3}
MODEL=${2:-runs/r1/best}
# ~ rather than /Users/<name>, SF2_ROM / SF2_MESEN from ~/.zshrc, and clear: the console shows no user name
CMD="clear && cd ~/${(q)${PWD#$HOME/}} && .venv/bin/python scripts/arcade.py --games $GAMES --model ${(q)MODEL}"
# the command goes in as an argument, not pasted into AppleScript source (paths have spaces and parentheses)
osascript -e 'on run argv' -e 'tell application "Terminal"' -e 'do script (item 1 of argv)' -e 'activate' \
    -e 'end tell' -e 'end run' "$CMD" >/dev/null
