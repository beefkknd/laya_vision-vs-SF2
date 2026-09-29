"""scripts/verify_replay.py re-plays sweep rows from their savestate recipe (gap, posture, action). Live-play rows
(posture "live", added to test_data/ by a4fe0db) have no such recipe: they must be left out of the sample and
counted, not crash the replay with KeyError: 'live'."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(HERE, "tests", "fixtures", "verify_replay", "rows.jsonl")


def verify_replay():
    if os.path.join(HERE, "scripts") not in sys.path:
        sys.path.insert(0, os.path.join(HERE, "scripts"))
    spec = importlib.util.spec_from_file_location("verify_replay", os.path.join(HERE, "scripts", "verify_replay.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_only_sweep_rows_are_replayed_and_live_rows_are_counted():
    recs = [json.loads(line) for line in open(FIXTURE)]
    keep, live = verify_replay().replayable(recs)
    assert [r["id"] for r in keep] == ["chunli-sweep_mid_g02-s.lp"]
    assert live == 1
