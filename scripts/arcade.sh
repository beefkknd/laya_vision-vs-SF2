#!/bin/zsh
# Open the arcade console in its own Terminal window: laya plays Chun-Li in N Mesen windows, forever.
#   scripts/arcade.sh                      # 3 games, runs/r1/best
#   scripts/arcade.sh 2 runs/r2/best       # games, model
# Ctrl-C in the console (or closing it) stops every game. Drag the windows wherever you like.
cd "$(dirname "$0")/.." || exit 1
GAMES=${1:-3}
MODEL=${2:-runs/r1/best}
: ${SF2_ROM:="$HOME/Downloads/sf2/Street Fighter II (USA).sfc"}
: ${SF2_MESEN:="$HOME/Applications/Mesen 2.app/Contents/MacOS/Mesen"}
CMD="cd ${(q)PWD} && export SF2_ROM=${(q)SF2_ROM} SF2_MESEN=${(q)SF2_MESEN} && .venv/bin/python scripts/arcade.py --games $GAMES --model ${(q)MODEL}"
osascript -e "tell application \"Terminal\" to do script \"${CMD//\"/\\\"}\"" -e 'tell application "Terminal" to activate' >/dev/null
