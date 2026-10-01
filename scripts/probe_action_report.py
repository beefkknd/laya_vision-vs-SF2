"""Report of the action-ID probe (sf2.data.action_probe): the attack-ID byte checked on pressed moves and in play.

    python scripts/probe_action_report.py --press <dir of probe_action_moves.py> --games <dir of probe_action_ram.py>
        [--json lessons/chunli_action_ids.json] [--sheets <dir>]

Prints (1) per pressed take of each character, on each player slot, the first attack IDs; whether player 1 and player 2
give the same ID sequence; (2) per character in play, the attack-state purity and the count of each ID; writes Chun-Li's
move <-> ID table as JSON and contact sheets (per ID, frames at the first row the ID is out, from different episodes).
"""
import argparse
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

import _path  # noqa: F401
from sf2.data import action_probe as A

ME = "chunli"


def load_press(d):
    takes = []
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        tag = os.path.basename(f)[:-4]
        meta, st, tk = json.loads(str(z["takes"])), z["struct"], z["take"]
        for i, t in enumerate(meta):
            off = (t["side"] - 1) * A.STRIDE
            takes.append(dict(t, file=tag, struct=st[tk == i][:, off:off + A.STRIDE]))
    return takes


def load_games(d):
    """(char, player, tag, struct rows, own shot flags) per fighter per game."""
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        tag = os.path.basename(f)[:-4]
        me, _, opp, _ = tag.split("_")
        names = list(z["names"])
        for p, ch in ((1, me), (2, opp)):
            shot = z["fields"][:, names.index("shot%d" % p)]
            out.append((ch, p, tag, z["struct"][:, (p - 1) * A.STRIDE:p * A.STRIDE], shot))
    return out


def press_report(takes):
    by = defaultdict(dict)
    for t in takes:
        by[(t["char"], t["action"], t["gap"] < 100)][t["side"]] = [e.ids for e in A.episodes(t["struct"]) if e.ids]
    same = sum(v.get(1) == v.get(2) for v in by.values() if len(v) == 2)
    both = sum(len(v) == 2 for v in by.values())
    print("pressed takes: player 1 vs player 2 attack-ID sequences equal in %d of %d (char, move, range)" % (same, both))
    for k, v in sorted(by.items()):
        if len(v) == 2 and v[1] != v[2]:
            print("  differs:", k, v)
    return same, both


def chunli_json(takes, games, path):
    mine = [t for t in takes if t["char"] == ME]
    by_move, by_id = A.move_table(mine)
    seen = Counter()
    for ch, p, tag, s, shot in games:
        if ch == ME:
            seen.update(A.id_counts(A.episodes(s, min_len=3)))
    during = defaultdict(set)        # every ID a pressed move's attack episodes step through -> the move
    for t in mine:
        for e in A.episodes(t["struct"]):
            for i in e.ids:
                during[i].add(t["action"])
    ids = {}
    for i in sorted(set(by_id) | set(seen) | set(during)):
        moves = by_id.get(i, [])
        base = sorted({m.split("@")[0] for m in moves})
        inside = sorted(during.get(i, ()))
        note = ("shared by %s" % ", ".join(base) if len(base) > 1 else "") if moves else (
            "a later phase of %s (its first ID is another; first only when an episode starts mid-move)" % ", ".join(inside) if inside else
            "not produced by any pressed move")
        ids[str(i)] = {"moves": moves, "seen_in_play": seen.get(i, 0), "in_sequence_of": inside, "note": note}
    no_id = sorted({"%s@%s" % (t["action"], "close" if t["gap"] < 100 else "far") for t in mine
                    if not any(e.ids for e in A.episodes(t["struct"]))})
    no_box = sorted({"%s@%s: %s" % (t["action"], "close" if t["gap"] < 100 else "far", A.no_box_kind(t["struct"], e))
                     for t in mine for e in A.episodes(t["struct"]) if e.state in A.ATTACK_STATES and not e.ids})
    rec = {
        "what": "Chun-Li's attack ID: the ROM's per-character move number of the attack box she has out",
        "address": {"player1": "0x%04X" % A.address(1, A.ATTACK_ID), "player2": "0x%04X" % A.address(2, A.ATTACK_ID),
                    "size": 1, "state_byte": {"player1": "0x0C03", "player2": "0x0E03"}},
        "rule": "an episode is a run of one state byte; its move is the FIRST non-zero attack ID in it (startup frames "
                "read 0; multi-hit moves step through several IDs). 0 all through = no attack box.",
        "measured": "scripts/probe_action_moves.py: VS BATTLE, Chun-Li vs a still Ken, each move pressed from the same "
                    "savestate at a close (26-28 px) and a far (151-153 px) gap, as player 1 and as player 2; "
                    "the move's inputs are sf2.data.vs_sweep.actions('chunli') (the controller's own) plus six "
                    "neutral-jump normals; seen_in_play: scripts/probe_action_ram.py games (runs/all8, explore 0.5)",
        "move_to_ids": by_move,
        "id_to_moves": ids,
        "moves_without_id": no_id,
        "attack_episodes_without_id": no_box,
        "shared_ids_explained": {
            "6": "a throw pressed out of grab range comes out as the far fierce (hp): the same real move",
            "7/8": "Lightning Legs starts with the short (lk) it is mashed from: a 0x0A lk episode, then the 0x0C "
                   "Lightning Legs episode (ID 50)"},
        "no_box_fallback": {"rule": "attack-state episode with no ID: own throw MOVE_CLASS (0x0CBA in 0x06 / 0x0C) "
                                    "-> throw; state 0x0C with SPECIAL_CLASS 0x0C49 -> that special; else cut (an "
                                    "attack stopped in its startup, 4-7 rows: unknown)",
                            "state 0x0C, 0x0C49 = 0x09": "spinning_bird_kick (46)",
                            "state 0x0C, 0x0C49 = 0x0C": "lightning_legs (50)",
                            "0x0CBA = 0x0C": "throw (close)"},
        "non_attack_moves": "walks, jumps, crouch, blocks, idle never put out an attack box (ID 0): name them by the "
                            "state byte (0x00 stand / walk by x, 0x02 crouch, 0x04 jump, 0x08 guard, 0x0E hit, "
                            "0x14 thrown), as sf2.data.movement does",
        "player_slots_agree": "same ID sequence on player 1 and player 2 (see the probe report)",
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=False)
        f.write("\n")
    return rec


def play_report(games):
    eps_by = defaultdict(list)
    for ch, p, tag, s, shot in games:
        for e in A.episodes(s, min_len=3):
            eps_by[ch].append((p, tag, e, s, shot))
    for ch, items in sorted(eps_by.items()):
        eps = [e for _, _, e, _, _ in items]
        pur = A.purity(eps)
        cnt = A.id_counts(eps)
        nobox = Counter(A.no_box_kind(s, e, shot) for p, tag, e, s, shot in items
                        if e.state in A.ATTACK_STATES and not e.ids)
        slots = {p: sorted(A.id_counts([e for q, _, e, _, _ in items if q == p])) for p in (1, 2)}
        print("%-8s attack episodes %d: with ID %.2f, one ID %.2f; no box: %s; %d IDs: %s" % (
            ch, pur["episodes"], pur["with_id"], pur["single_id"], dict(nobox), len(cnt),
            " ".join("%d:%d" % kv for kv in sorted(cnt.items()))))
        if all(slots.values()):
            print("          IDs as player 1 %s / as player 2 %s" % (slots[1], slots[2]))
    return eps_by


def sheets(eps_by, games_dir, out_dir, chars=("ken", ME), per_id=3, seed=0):
    from PIL import Image, ImageDraw

    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    paths = []
    for ch in chars:
        groups = defaultdict(list)
        for p, tag, e, s, shot in eps_by.get(ch, []):
            if e.attack_id and e.state in A.ATTACK_STATES + (A.JUMP_STATE,):
                first = e.start + int(np.flatnonzero(s[e.start:e.end, A.ATTACK_ID])[0])
                groups[e.attack_id].append((p, tag, first))
        ids = sorted(groups)
        w, h, cols = 192, 168, per_id
        sheet = Image.new("RGB", (cols * w, len(ids) * (h + 14)), "white")
        d = ImageDraw.Draw(sheet)
        for r, i in enumerate(ids):
            picks = rng.sample(groups[i], min(per_id, len(groups[i])))
            for c, (p, tag, row) in enumerate(picks):
                path = os.path.join(games_dir, "images", tag, "k%05d.png" % (row + 1))   # capture k shows row k - 1
                if not os.path.exists(path):
                    continue
                im = Image.open(path).convert("RGB").resize((w, h))
                sheet.paste(im, (c * w, r * (h + 14) + 14))
                d.text((c * w + 2, r * (h + 14)), "%s act%02d  P%d %s r%d (n=%d)" % (ch, i, p, tag.split("_vs_")[1],
                                                                                    row, len(groups[i])), fill="black")
        path = os.path.join(out_dir, "action_ids_%s.png" % ch)
        sheet.save(path)
        paths.append(path)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--press", required=True)
    ap.add_argument("--games", required=True)
    ap.add_argument("--json", default=os.path.join("lessons", "chunli_action_ids.json"))
    ap.add_argument("--sheets", default=None)
    args = ap.parse_args()
    takes = load_press(args.press)
    games = load_games(args.games)
    press_report(takes)
    rec = chunli_json(takes, games, args.json)
    print("Chun-Li: move -> IDs", json.dumps(rec["move_to_ids"]))
    print("Chun-Li: moves with no ID", rec["moves_without_id"])
    eps_by = play_report(games)
    if args.sheets:
        print("sheets:", sheets(eps_by, args.games, args.sheets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
