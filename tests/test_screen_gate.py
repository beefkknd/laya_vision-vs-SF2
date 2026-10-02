"""The gate's own compare (scripts/gate_screen_reader.py): a reader output equal to the referee passes every bar; each
known-bad reader output (x off, swapped identity, late round over, wrong health / air / facing / actions) fails the
bar it breaks. Synthetic games, no files."""
import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import gate_screen_reader as G  # noqa: E402

N, RK = 200, 180
MAJORITY = {"ryu/a": "stand", "ryu/b": "jump", "ken/a": "stand", "ken/b": "attack"}
AIR = {"ryu/a": "ground", "ryu/b": "air", "ken/a": "ground", "ken/b": "ground"}


def _truth():
    frames = []
    for k in range(N):
        jump = 50 <= k < 80
        fr = dict(k=k, gap=100 if k < 150 else 20, shot=False, shot0=False, shot_drawn=False)
        fr[1] = dict(x=60 + k % 7, x_lag1=60, air=jump, air0=jump, facing="right", facing0="right", life=0.5, hp=0.5,
                     hp0=0.5, act="jump" if jump else "stand", key="ryu/b" if jump else "ryu/a", drawn="right")
        fr[2] = dict(x=160, x_lag1=160, air=False, air0=False, facing="left", facing0="left", life=0.75, hp=0.75,
                     hp0=0.75, act="attack" if k % 10 == 0 else "stand", key="ken/b" if k % 10 == 0 else "ken/a",
                     drawn="left")
        frames.append(fr)
    return dict(meta=dict(p1="ryu", p2="ken", name="g", set="a", stage="ryu"), frames=frames, result_k=RK)


def _perfect(truth):
    out = []
    for f in truth["frames"][1:]:
        per = {}
        for p in (1, 2):
            t = f[p]
            per[p] = dict(found=True, x=t["x"], air=t["air"], facing=t["facing"], sprite=t["key"],
                          action=MAJORITY[t["key"]], unknown=False, conf=1.0, health=t["life"],
                          side="left" if p == 1 else "right")
        out.append(dict(k=f["k"], p=per, over=f["k"] >= RK + G.LAG, proj=0, proj_sides=[], timer=50))
    return dict(frames=out, locks=[dict(k=1, chars=["ryu", "ken"])])


def _bars(reader):
    s = G.summarize(G.merge([G.compare(_truth(), reader, MAJORITY, AIR)]))
    return G.bars(s, 5.0)


def _break(fn):
    r = _perfect(_truth())
    r = copy.deepcopy(r)
    fn(r)
    return _bars(r)


def test_perfect_reader_passes_every_bar():
    b = _bars(_perfect(_truth()))
    assert all(v["ok"] for v in b.values()), b


@pytest.mark.parametrize("name,fault", [
    ("x", lambda r: [f["p"][1].update(x=f["p"][1]["x"] + 10) for f in r["frames"]]),
    ("identity", lambda r: r["locks"][0].update(chars=["ken", "ryu"])),
    ("round_over", lambda r: [f.update(over=f["k"] >= RK + G.LAG + 20) for f in r["frames"]]),
    ("round_over", lambda r: [f.update(over=False) for f in r["frames"]]),
    ("health", lambda r: [f["p"][2].update(health=0.70) for f in r["frames"]]),
    ("air", lambda r: [f["p"][1].update(air=False) for f in r["frames"]]),
    ("facing", lambda r: [f["p"][2].update(facing="right") for f in r["frames"][::5]]),
    ("action", lambda r: [f["p"][p].update(action=None, unknown=True) for f in r["frames"][::4] for p in (1, 2)]),
])
def test_known_bad_reader_fails_its_bar(name, fault):
    b = _break(fault)
    assert not b[name]["ok"], b[name]


def test_x_bar_ignores_near_contact_frames():
    """A wrong x only while the fighters are in contact (gap < 40) does not fail the non-overlap x bar."""
    b = _break(lambda r: [f["p"][1].update(x=999) for f in r["frames"] if f["k"] >= 150])
    assert b["x"]["ok"]


def test_speed_bar():
    s = G.summarize(G.merge([G.compare(_truth(), _perfect(_truth()), MAJORITY, AIR)]))
    assert not G.bars(s, 25.0)["speed_ms"]["ok"] and G.bars(s, 19.9)["speed_ms"]["ok"]
