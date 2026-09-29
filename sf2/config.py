"""Constants shared by every script. Change them here, not in the scripts."""
import os

# The checkout: every path into the repo is built from here (modules never count folders up from __file__).
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def _env(name: str, default: str) -> str:
    """A machine-specific setting: ``$name`` when set, else ``default``."""
    return os.path.expanduser(os.environ.get(name, default))


# Emulator: Mesen 2 running SNES Street Fighter II, driven over a local socket by mesen/sf2_bridge.lua.
MESEN_APP = _env("SF2_MESEN", "/Applications/Mesen.app/Contents/MacOS/Mesen")
# Also tried: ~/Applications, where a per-user drag install lands (it may keep its "Mesen 2" download name).
MESEN_CANDIDATES = [MESEN_APP] + [os.path.expanduser("~/Applications/%s.app/Contents/MacOS/Mesen" % app)
                                  for app in ("Mesen", "Mesen 2")]
MESEN_SETTINGS = _env("SF2_MESEN_SETTINGS", "~/Library/Application Support/MesenCE/settings.json")
# Ports: (first, how many). A runner that fans out uses first .. first + how many - 1 (tests/test_layout.py checks
# that no two ranges overlap, so runners can play side by side).
PORTS = {"mesen": (47800, 1), "learn": (47990, 1), "vs_moves": (47991, 1), "dataset": (48001, 128),
         "replay": (48401, 32), "system1": (48901, 16), "ab": (49101, 64), "notebook": (49501, 2),
         "panel": (8765, 1)}
MESEN_PORT = PORTS["mesen"][0]
# The ROM: Street Fighter II (USA), SHA1 7DDCB96E0D9FEA94D9370635262AC7C28DA85214 (git-ignored; $SF2_ROM overrides)
DEFAULT_ROM = "roms/Street Fighter II (USA).sfc"

# System 2: Qwen 3.8 27B (Q4, MTP) served by omlx on this machine (OpenAI-style chat API); start it with
# ~/work/omlx/start. Thinking is OFF and the reply capped: with it on, one playbook review spent 11,593 tokens
# (331 s) reasoning for a 1,177-character answer (the same lesson as agent_harness's model_client.py).
QWEN_URL = _env("SF2_QWEN_URL", "http://127.0.0.1:8000/v1/chat/completions")
QWEN_MODEL = _env("SF2_QWEN_MODEL", "Jundot--Qwen3.8-27B-oQ4e-mtp")
QWEN_THINKING = False
QWEN_MAX_TOKENS = 8192
QWEN_TEMPERATURE = 0.0

# laya-vision: SmolVLM-256M backbone + Laya typed-decision head (PyTorch, runs on Apple MPS).
BASE_MODEL = "thaitea/laya-vision-smolvlm-256m"
# What laya-vision sees: 256x256 frames at their native pixels. sf2/data/frames.py pads the 256x224 screen to 256x256,
# and these settings make laya's image prep an exact identity at that size (checked: max pixel difference 0). The
# base checkpoint's own settings (image_size 512 via the processor's 2048 px LANCZOS hop) would upscale and resample.
IMAGE_SIZE = 256
IMAGE_CFG = {"image_size": IMAGE_SIZE, "preprocess": "gpu", "image_interpolation": "nearest"}
LAYA_VISION_REPO = "https://github.com/r33drichards/laya-vision"
LAYA_VISION_COMMIT = "568feeeada793f70f736756b0f3a7643d1e75910"

HOLD = 4               # frames between two decisions (the dataset's rhythm)

# SNES pad, SF2's default layout: Y X L = jab / strong / fierce punch, B A R = short / forward / roundhouse.
# Names are Mesen's (emu.getInput / emu.setInput keys).
PAD = {"lp": "y", "mp": "x", "hp": "l", "lk": "b", "mk": "a", "hk": "r"}

# The models System 1 plays with (checkpoint dirs under runs/, git-ignored).
LAYA_VISION = _env("SF2_LAYA_VISION", "runs/all8/best")
TEXT_LAYA = _env("SF2_TEXT_LAYA", "runs/text_laya/advice_v1")
TEXT_LAYA_BASE = "aac6fef/laya-mlx"      # text laya: the MLX conversion of convaiinnovations/laya
# Text laya runs in the laya-mlx venv (Apple MLX), not this project's torch venv; its weights in the HF cache.
MLX_PYTHON = _env("SF2_MLX_PYTHON", "~/work/laya_mlx/.venv/bin/python")
HF_HOME = _env("HF_HOME", "/Volumes/ExtremeSSD/huggingface")

# Folders the loop and the demo write (all git-ignored).
MEMORY_RUNS = "memory_runs"              # learn_loop --fresh NAME: memory_runs/NAME
MEMORY_SEED = "memory_seeds/video"       # the brain panel's "cheat" copies from here
LIVE = "out/live"                        # what the running loop shows the brain panel
VIDEO_OUT = _env("SF2_VIDEO_OUT", "~/Desktop/laya_video")
