"""Tables and checks over the built catalog (catalog.json): per-character counts, the shared-sprite table, the
re-render check, contact sheets."""
import itertools
import json
import os
import random
from collections import Counter
from typing import Dict, List

import numpy as np
from PIL import Image

from .catalog import canonical, key_of
from .collect import oam_facing
from .oam import Snapshot, render_groups

PROJ = "_projectiles"


def per_char(sprites: Dict[str, Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for e in sprites.values():
        d = out.setdefault(e["char"], dict(sprites=0, frames=0, index_keys=set(), facing_mismatch=0))
        d["sprites"] += 1
        d["frames"] += e["frames"]
        d["index_keys"].add(e["index_key"])
        d["facing_mismatch"] += e.get("facing_mismatch", 0)
    for d in out.values():
        d["index_keys"] = len(d["index_keys"])
    return out


def shared(sprites: Dict[str, Dict], field: str = "labels", ignore=("unknown",)) -> Dict:
    """Sprites whose 7-answer action (``field`` = labels: the lag used, lag0: lag 0) takes more than one value.
    Per character: shared sprites, their frames and the share of all frames; per action pair: sprites and frames,
    plus the minority frames (frames of a shared sprite NOT under its majority answer: the frames a sprite->action
    lookup would get wrong)."""
    chars: Dict[str, Counter] = {}
    pairs: Dict[tuple, Counter] = {}
    for e in sprites.values():
        if e["char"] == PROJ:
            continue
        acts = {a: n for a, n in e[field]["act2"].items() if a not in ignore}
        c = chars.setdefault(e["char"], Counter())
        c["frames"] += sum(acts.values())
        c["sprites"] += 1
        if len(acts) < 2:
            continue
        c["shared"] += 1
        c["shared_frames"] += sum(acts.values())
        c["minority_frames"] += sum(acts.values()) - max(acts.values())
        for x, y in itertools.combinations(sorted(acts), 2):
            p = pairs.setdefault((x, y), Counter())
            p["sprites"] += 1
            p["frames"] += acts[x] + acts[y]
            p["minor"] += min(acts[x], acts[y])
    return dict(chars=chars, pairs=pairs)


def load_snap(path: str):
    z = np.load(path)
    mode, base, off = map(int, z["regs"])
    return Snapshot(z["oam"].tobytes(), z["cg"].tobytes(), z["vram"].tobytes(), mode, base, off), int(z["pal"])


def rerender_check(out: str, sprites: Dict[str, Dict], n: int = 50, seed: int = 0) -> List[str]:
    """For ``n`` random catalog entries: re-render from the stored snapshot, canonicalise, and compare key and
    pixels with the catalog PNG. The failures (empty = all good)."""
    rng = random.Random(seed)
    bad = []
    for ck in rng.sample(sorted(sprites), min(n, len(sprites))):
        e = sprites[ck]
        snap, pal = load_snap(os.path.join(out, "_pairs", e["snap"], "snaps", e["key"] + ".npz"))
        g = render_groups(snap, pals={pal})[pal]
        rgba = canonical(g.rgba, oam_facing(snap, pal))
        png = np.asarray(Image.open(os.path.join(out, ck + ".png")).convert("RGBA"))
        if key_of(rgba) != e["key"] or rgba.shape != png.shape or not np.array_equal(rgba, png):
            bad.append(ck)
    return bad


def contact_sheet(out: str, sprites: Dict[str, Dict], char: str, n: int = 40, seed: int = 0, cell: int = 128,
                  cols: int = 8) -> str:
    """``n`` random sprites of ``char`` on a grey checker, each with its majority action written under it."""
    from PIL import ImageDraw
    rng = random.Random(seed)
    keys = [ck for ck, e in sprites.items() if e["char"] == char]
    if not keys:
        return ""
    keys = rng.sample(sorted(keys), min(n, len(keys)))
    rows = (len(keys) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * cell, rows * (cell + 12)), (90, 90, 90, 255))
    draw = ImageDraw.Draw(sheet)
    for j, ck in enumerate(keys):
        im = Image.open(os.path.join(out, ck + ".png")).convert("RGBA")
        s = min(1.0, (cell - 4) / max(im.size))
        if s < 1:
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.NEAREST)
        x, y = (j % cols) * cell, (j // cols) * (cell + 12)
        sheet.alpha_composite(im, (x + (cell - im.width) // 2, y + cell - im.height))
        acts = sprites[ck]["labels"]["act2"] if "labels" in sprites[ck] else {}
        top = max(acts, key=acts.get) if acts else "?"
        draw.text((x + 2, y + cell), "%s%s" % (top, "*" if len(acts) > 1 else ""), fill=(255, 255, 0, 255))
    path = os.path.join(out, "_sheets", "%s.png" % char.strip("_"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.convert("RGB").save(path)
    return path


def load(out: str) -> Dict:
    with open(os.path.join(out, "catalog.json")) as f:
        return json.load(f)
