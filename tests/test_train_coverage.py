"""What scripts/train.py trains on must represent every character: run the REAL test_data/ through the same
load + split + coverage gate train.py uses (sf2/data/train_data.py). The old global split, which left Ken and Dhalsim
with no validation rows, is kept below as a known-bad twin that must fail the gate."""
import os
import random

import pytest

from sf2.data.train_data import coverage_problems, load_data, position, split_by_position
from sf2.vocab import FIGHTERS

CHARS = FIGHTERS
DIRS = [os.path.join("test_data", c) for c in CHARS]
real = pytest.mark.skipif(not all(os.path.exists(os.path.join(d, "train.jsonl")) for d in DIRS),
                          reason="test_data/ not built (scripts/vs_dataset.py run)")


@pytest.fixture(scope="module")
def rows():
    return load_data(DIRS, seed=0)


def _old_global_split(rows, rng, share=0.05):
    """The split that shipped first: positions sampled across all characters at once."""
    keys = sorted({position(ex) for ex in rows})
    held = set(rng.sample(keys, max(1, round(share * len(keys)))))
    return [ex for ex in rows if position(ex) not in held], [ex for ex in rows if position(ex) in held]


@real
def test_real_data_passes_the_gate(rows):
    train, val = rows
    assert coverage_problems(train, val, DIRS) == []


@real
def test_every_character_in_train_and_val_equally(rows):
    train, val = rows
    for split in (train, val):
        n = {c: sum(ex["dataset"] == c for ex in split) for c in CHARS}
        assert min(n.values()) > 0 and min(n.values()) == max(n.values()), n


@real
def test_old_global_split_fails_the_gate(rows):
    train, val = rows
    tr, va = _old_global_split(train + val, random.Random(0))
    assert any("val rows unequal" in p for p in coverage_problems(tr, va, DIRS))
    # the original failure, a character with no validation rows at all, is named by the gate
    no_dhalsim = [ex for ex in va if ex["dataset"] != "dhalsim"]
    assert "dhalsim has no val rows" in coverage_problems(tr, no_dhalsim, DIRS)


@real
def test_a_missing_character_fails_the_gate(rows):
    train, val = rows
    problems = coverage_problems([ex for ex in train if ex["dataset"] != "honda"], val, DIRS)
    assert "honda has no train rows" in problems


@real
def test_a_thin_combination_fails_the_gate(rows):
    import json
    ids = {json.loads(line)["id"] for line in open("test_data/ken/train.jsonl")
           if json.loads(line)["action"] == "hadoken" and json.loads(line)["range"] == "far"}
    train, val = rows
    problems = coverage_problems([ex for ex in train if not (ex["dataset"] == "ken" and ex["id"] in ids)], val, DIRS)
    assert any(p.startswith("ken hadoken@far/") for p in problems), problems[:5]


def test_split_never_shares_a_position_and_is_per_dataset():
    rows = [{"dataset": c, "state": {"images": ["frames/%s%s_%d_now.png" % (m, s, i)]}}
            for c in ("ryu", "ken") for s in ("left_close", "left_far") for i in range(10)
            for a in range(20) for m in ("", "mirror_")]
    tr, va = split_by_position(rows, random.Random(0))
    assert not {position(x) for x in tr} & {position(x) for x in va}
    per = {c: len({position(x) for x in va if x["dataset"] == c}) for c in ("ryu", "ken")}
    assert per["ryu"] == per["ken"] >= 1
