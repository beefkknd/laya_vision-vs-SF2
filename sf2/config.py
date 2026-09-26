"""Constants shared by every script. Change them here, not in the scripts."""

# Emulator: Mesen 2 running SNES Street Fighter II, driven over a local socket by mesen/sf2_bridge.lua.
MESEN_PORT = 47800
# RAM addresses differ per cartridge (World Warrior / Turbo / Super SF2, region). Find yours with
# scripts/find_ram.py; it writes this file.
DEFAULT_RAM_MAP = "ram_maps/sf2_snes.txt"
# A savestate at the start of a fight, made with scripts/record_human.py (press F9 in Mesen).
DEFAULT_SAVESTATE = "states/ryu_vs_ken.state"

# laya-vision: SmolVLM-256M backbone + Laya typed-decision head (PyTorch, runs on Apple MPS).
BASE_MODEL = "thaitea/laya-vision-smolvlm-256m"
LAYA_VISION_REPO = "https://github.com/r33drichards/laya-vision"
LAYA_VISION_COMMIT = "568feeeada793f70f736756b0f3a7643d1e75910"

FPS = 60
HOLD = 4               # frames a non-macro action is held => one decision every 4 frames
PREV_GAP = 4           # the "previous" image is always the frame HOLD frames before the current one
NEXT_WINDOW = 30       # 0.5 s: window for damage_for / damage_against after a decision
ROUND_LIFE = 176       # full health in the verified Street Fighter II (USA) RAM map

# SNES pad, SF2's default layout: Y X L = jab / strong / fierce punch, B A R = short / forward / roundhouse.
# Names are Mesen's (emu.getInput / emu.setInput keys).
PAD = {"lp": "y", "mp": "x", "hp": "l", "lk": "b", "mk": "a", "hk": "r"}
BUTTONS = ["a", "b", "x", "y", "l", "r", "up", "down", "left", "right", "select", "start"]
