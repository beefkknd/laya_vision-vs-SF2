"""Constants shared by every script. Change them here, not in the scripts."""

# stable-retro 1.0.x integration name (older gym-retro / stable-retro 0.9 drop the "-v0").
GAME = "StreetFighterIISpecialChampionEdition-Genesis-v0"
# The only savestate that ships with the integration. Make your own (Ryu vs Ken, training mode, ...) with
# scripts/record_human.py --save-state-to states/ryu_vs_ken.state and pass --state states/ryu_vs_ken.state.
DEFAULT_STATE = "Champion.Level1.RyuVsGuile"

# laya-vision: SmolVLM-256M backbone + Laya typed-decision head (PyTorch, runs on Apple MPS).
BASE_MODEL = "thaitea/laya-vision-smolvlm-256m"
LAYA_VISION_REPO = "https://github.com/r33drichards/laya-vision"
LAYA_VISION_COMMIT = "568feeeada793f70f736756b0f3a7643d1e75910"

FULL_HP = 176          # SF2 SCE Genesis life bar; -1 on KO
FPS = 60
HOLD = 4               # frames a non-macro action is held => one decision every 4 frames
PREV_GAP = 4           # the "previous" image is always the frame HOLD frames before the current one
NEXT_WINDOW = 30       # 0.5 s: window for damage_for / damage_against after a decision
WHIFF_WINDOW = 60      # a hadouken that deals no damage within 1 s counts as a whiff

# Genesis 6-button pad in SF2 SCE: top row X Y Z = light / medium / fierce punch, bottom row A B C = kicks.
PAD = {"lp": "X", "mp": "Y", "hp": "Z", "lk": "A", "mk": "B", "hk": "C"}
