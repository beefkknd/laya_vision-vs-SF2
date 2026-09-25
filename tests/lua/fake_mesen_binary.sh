#!/usr/bin/env bash
# Stands in for "Mesen --testrunner <rom> <script.lua>": runs the script under the Lua mock of Mesen.
# The "ROM" argument is used as the PNG the mock returns for every screenshot.
[ "$1" = "--testrunner" ] || { echo "expected --testrunner" >&2; exit 2; }
exec lua5.4 "$(dirname "$0")/mock_mesen.lua" "$3" "$2"
