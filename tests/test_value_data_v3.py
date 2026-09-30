"""The value dataset builder's opt-in flags for the second run (docs/reviews/2026-09-30_dr_fable_lv_value.md B.2-3):
``no_cap`` (every character keeps all its new training games; no test_extra) and ``forward_value_cap="median_attack"``
(the new live forward VALUE training rows subsampled per character to the median attack's count, by a hash of the
row id; the mirror follows its source). The defaults must stay byte for byte the first run's build (golden digests
taken from the code before these flags existed). Tiny fixtures in tmp_path only."""
import collections
import hashlib
import os
import subprocess
import sys

import pytest

from sf2.data import value_data as V
from test_value_data import REPO, build, by_id, entry, load, make_new, make_old, write_log

# sha256 over (relative path, bytes) of every file of the fixture build, taken with the code before --no-cap and
# --forward-value-cap existed (commit 3c833d4): the defaults must reproduce the first run's data exactly
GOLDEN = {"plain": "ed4cfab8b57ec5d6716eb47b6747f3f945f38cf795a52d9fdd0dc4b760e55cf1",
          "capped": "79f4e2fd7db9ab7a15cbf65413a2091c86c5746df0a7e2cda01058a76fe719d3"}
ATTACKS = ['lp', 'mp', 'hp', 'lk', 'mk', 'hk', 'c.lk', 'c.mk', 'sweep', 'c.hp', 'throw', 'lightning_legs',
           'spinning_bird_kick']


def digest(root):
    h = hashlib.sha256()
    for d, _, files in sorted(os.walk(root)):
        for f in sorted(files):
            p = os.path.join(d, f)
            h.update(os.path.relpath(p, root).encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def add_ryu(repo):
    """Ryu vs Ken, 2 training decisions: the cap (2) moves Chun-Li's game 1 vs Ryu to test_extra by default."""
    write_log(repo, "p_vs_ryu", "ryu", [entry(0, 0, "hk", opp="ken", me="ryu"), entry(1, 0, "hk", opp="ken", me="ryu")])


@pytest.fixture
def repo(tmp_path):
    make_old(str(tmp_path))
    make_new(str(tmp_path))
    return str(tmp_path)


# --- defaults unchanged ---

def test_default_build_is_byte_identical_to_the_first_runs(repo):
    assert build(repo)["problems"] == []
    assert digest(os.path.join(repo, "test_data_v2")) == GOLDEN["plain"]


def test_default_build_with_the_cap_engaged_is_byte_identical(repo):
    add_ryu(repo)
    assert build(repo)["cap"] == 2
    assert digest(os.path.join(repo, "test_data_v2")) == GOLDEN["capped"]


def test_explicit_defaults_are_the_defaults(repo):
    add_ryu(repo)
    build(repo, no_cap=False, forward_value_cap=None)
    assert digest(os.path.join(repo, "test_data_v2")) == GOLDEN["capped"]


# --- --no-cap ---

def test_no_cap_keeps_every_training_game_and_writes_no_test_extra(repo):
    add_ryu(repo)
    s = build(repo, no_cap=True)
    assert s["problems"] == [] and s["cap"] is None
    assert s["counts"]["chunli"]["new_train_decisions"] == 3 and s["counts"]["chunli"]["extra_decisions"] == 0
    assert s["counts"]["ryu"]["new_train_decisions"] == 2
    train = by_id(load(repo, "train"))
    moved = "chunli-live_lv_ryu_g01_f00000-block_high"          # the capped build sends it to test_extra
    assert {moved, moved + "-m", moved + "-value", moved + "-m-value"} <= set(train)
    for char in ("chunli", "ryu"):
        assert not os.path.exists(os.path.join(repo, "test_data_v2", char, V.EXTRA_FILE + ".jsonl"))


def test_no_cap_leaves_test_games_and_the_holdout_unchanged(tmp_path):
    for flag in (False, True):
        r = str(tmp_path / str(flag))
        make_old(r)
        make_new(r)
        add_ryu(r)
        build(r, no_cap=flag)
    for name in ("test_real_left", "test_real_right", V.HELDOUT_FILE):
        assert load(str(tmp_path / "False"), name) == load(str(tmp_path / "True"), name), name
    assert not [r for r in load(str(tmp_path / "True"), "train") if r.get("opp") == "guile"]


def test_script_no_cap_flag(repo):
    add_ryu(repo)
    out = os.path.join(repo, "v3")
    p = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "build_value_data.py"), "--test-data",
                        os.path.join(repo, "test_data"), "--lv-root", os.path.join(repo, "rollouts/lv_value"),
                        "--repo", repo, "--out", out, "--min-value-train", "0", "--no-cap"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "live cap: off" in p.stdout
    assert not os.path.exists(os.path.join(out, "chunli", V.EXTRA_FILE + ".jsonl"))


# --- --forward-value-cap median_attack ---

def test_median_attack_counts_every_attack_of_choices_zero_included():
    counts = {a: 3 for a in ATTACKS[:7]}                  # 7 attacks with 3 rows, 6 with none: median 3
    assert V.median_attack(counts, ATTACKS) == 3
    assert V.median_attack({a: 3 for a in ATTACKS[:6]}, ATTACKS) == 0   # 6 of 13: the median attack has none
    assert V.median_attack({"lp": 4, "mp": 7}, ["lp", "mp"]) == 5       # an even count: the mean, rounded down


def test_forward_keep_is_a_deterministic_hash_subsample():
    ids = ["chunli-live_lv_ryu_g%02d_f00000-forward" % g for g in range(10)]
    kept = V.forward_value_keep(ids, 4)
    assert len(kept) == 4 and kept <= set(ids)
    assert kept == V.forward_value_keep(list(reversed(ids)), 4)          # order of input does not matter
    assert V.forward_value_keep(ids, 4) <= V.forward_value_keep(ids, 6)   # nested: more keeps a superset
    assert V.forward_value_keep(ids, 20) == set(ids)
    import random
    random.seed(1)
    again = V.forward_value_keep(ids, 4)
    random.seed(2)
    assert again == V.forward_value_keep(ids, 4) == kept               # no random state involved


def forward_repo(repo):
    """Chun-Li vs Ryu, training games 0, 1, 3, 4, 6: 7 attacks x 3 decisions, 12 forwards (and a blocked move)."""
    games = [0, 1, 3, 4, 6]
    rows = [entry(games[i % 5], 1000 + 10 * i + k, a, dealt=5 * k) for k, a in enumerate(ATTACKS[:7]) for i in range(3)]
    rows += [entry(games[i % 5], 5000 + 10 * i, "forward", taken=i % 3) for i in range(12)]
    rows += [entry(7, 9000, "block_low", taken=12)]
    write_log(repo, "fwd", "chunli", rows)


def new_value(rows, action, mirrored=None):
    return [r for r in rows if r.get("task") == "value" and r["action"] == action and "-live_lv_" in r["id"]
            and (mirrored is None or r["mirrored"] == mirrored)]


def test_forward_value_rows_capped_to_the_median_attack(repo):
    forward_repo(repo)
    s = build(repo, forward_value_cap="median_attack")
    assert s["problems"] == []
    train = load(repo, "train")
    # the 13 attacks' new value training rows: 0 x5, throw 1, six at 3, c.lk 4 (+ make_new's): median 3
    fc = s["counts"]["chunli"]["forward_value_cap"]
    assert fc["median_attack"] == 3 and fc["before"] == 12 and fc["after"] == 3
    assert len(new_value(train, "forward", mirrored=False)) == 3
    assert len(new_value(train, "forward", mirrored=True)) == 3
    real = {r["id"] for r in new_value(train, "forward", mirrored=False)}
    mir = {r["id"] for r in new_value(train, "forward", mirrored=True)}
    assert {i.replace("-value", "-m-value") for i in real} == mir                  # the mirror follows its source
    outcome_fwd = [r for r in train if r["action"] == "forward" and "-live_lv_" in r["id"] and r.get("task") != "value"]
    assert len(outcome_fwd) == 24                                                  # outcome rows untouched (+ mirrors)
    assert sum(fc["labels_before"].values()) == 12 and sum(fc["labels_after"].values()) == 3


def test_forward_cap_leaves_other_moves_and_other_files_untouched(tmp_path):
    for flag in (None, "median_attack"):
        r = str(tmp_path / str(flag))
        make_old(r)
        make_new(r)
        forward_repo(r)
        build(r, forward_value_cap=flag)
    a, b = str(tmp_path / "None"), str(tmp_path / "median_attack")
    plain, capped = load(a, "train"), load(b, "train")
    not_fwd_value = lambda rows: [x for x in rows if not (x.get("task") == "value" and x["action"] == "forward"
                                                          and "-live_lv_" in x["id"])]
    assert not_fwd_value(plain) == not_fwd_value(capped)
    for name in ("test_real_left", "test_real_right", V.HELDOUT_FILE, V.EXTRA_FILE):
        assert load(a, name) == load(b, name), name


def test_forward_cap_is_deterministic(tmp_path):
    for n in ("a", "b"):
        r = str(tmp_path / n)
        make_old(r)
        make_new(r)
        forward_repo(r)
        build(r, forward_value_cap="median_attack")
    assert digest(str(tmp_path / "a" / "test_data_v2")) == digest(str(tmp_path / "b" / "test_data_v2"))


def test_forward_under_the_median_is_untouched(repo):
    s = build(repo, forward_value_cap="median_attack")      # make_new has no forward at all
    fc = s["counts"]["chunli"]["forward_value_cap"]
    assert fc["before"] == 0 and fc["after"] == 0


def test_unknown_forward_value_cap_is_refused(repo):
    with pytest.raises(ValueError):
        build(repo, forward_value_cap="mean")


def test_script_forward_value_cap_flag(repo):
    forward_repo(repo)
    out = os.path.join(repo, "v3")
    p = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "build_value_data.py"), "--test-data",
                        os.path.join(repo, "test_data"), "--lv-root", os.path.join(repo, "rollouts/lv_value"),
                        "--repo", repo, "--out", out, "--min-value-train", "0", "--no-cap",
                        "--forward-value-cap", "median_attack"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "forward value rows 12 -> 3" in p.stdout
    labels = collections.Counter(r["action"] for r in load(repo, "train", out="v3")
                                 if r.get("task") == "value" and not r["mirrored"] and "-live_lv_" in r["id"])
    assert labels["forward"] == 3
