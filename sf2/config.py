"""Constants shared by every script. Change them here, not in the scripts."""

# Emulator: Mesen 2 running SNES Street Fighter II, driven over a local socket by mesen/sf2_bridge.lua.
MESEN_PORT = 47800
# The ROM: Street Fighter II (USA), SHA1 7DDCB96E0D9FEA94D9370635262AC7C28DA85214 (git-ignored; $SF2_ROM overrides)
DEFAULT_ROM = "roms/Street Fighter II (USA).sfc"

# System 2: Qwen 3.8 27B (Q4, MTP) served by omlx on this machine (OpenAI-style chat API); start it with
# ~/work/omlx/start. Thinking is OFF and the reply capped: with it on, one playbook review spent 11,593 tokens
# (331 s) reasoning for a 1,177-character answer (the same lesson as agent_harness's model_client.py).
QWEN_URL = "http://127.0.0.1:8000/v1/chat/completions"
QWEN_MODEL = "Jundot--Qwen3.8-27B-oQ4e-mtp"
QWEN_THINKING = False
QWEN_MAX_TOKENS = 8192
QWEN_TEMPERATURE = 0.0

# laya-vision: SmolVLM-256M backbone + Laya typed-decision head (PyTorch, runs on Apple MPS).
BASE_MODEL = "thaitea/laya-vision-smolvlm-256m"
# What laya-vision sees: 256x256 frames at their native pixels. sf2/frames.py pads the 256x224 screen to 256x256,
# and these settings make laya's image prep an exact identity at that size (checked: max pixel difference 0). The
# base checkpoint's own settings (image_size 512 via the processor's 2048 px LANCZOS hop) would upscale and resample.
IMAGE_SIZE = 256
IMAGE_CFG = {"image_size": IMAGE_SIZE, "preprocess": "gpu", "image_interpolation": "nearest"}
LAYA_VISION_REPO = "https://github.com/r33drichards/laya-vision"
LAYA_VISION_COMMIT = "568feeeada793f70f736756b0f3a7643d1e75910"

HOLD = 4               # frames between two decisions (the dataset's rhythm)
PREV_GAP = 4           # the "previous" image is always the frame HOLD frames before the current one

# SNES pad, SF2's default layout: Y X L = jab / strong / fierce punch, B A R = short / forward / roundhouse.
# Names are Mesen's (emu.getInput / emu.setInput keys).
PAD = {"lp": "y", "mp": "x", "hp": "l", "lk": "b", "mk": "a", "hk": "r"}
