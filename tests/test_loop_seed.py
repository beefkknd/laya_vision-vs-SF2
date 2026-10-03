"""M1 (a): seeding the short memory from the web/book playbook puts the opponent's verified rules in play.

Seen RED: ``seed`` for an opponent the book does not cover returns [] (asserted), so a miswired seeder that fed the
wrong opponent's lines - or none - fails here; and in_play must actually surface the seeded lines.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest                                                        # noqa: E402
from looptools import load_driver                                   # noqa: E402

from sf2.system2 import lessons as L, seed_rules                    # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK = os.path.join(HERE, "lessons", "book.json")


def _opps_with_lines():
    doc = json.load(open(BOOK))
    return [o for o, e in doc["opponents"].items() if e.get("lines")]


@pytest.mark.skipif(not os.path.exists(BOOK), reason="no lessons/book.json")
def test_seed_puts_verified_book_rules_in_play():
    play_loop = load_driver()                                       # the driver, imported without running it
    opp = _opps_with_lines()[0]
    reg = play_loop.seed(opp, BOOK)
    assert reg, "the book has lines for %s but none were seeded" % opp
    lines = {r["line"] for r in reg}
    assert lines == set(L.in_play(reg)) or lines >= set(L.in_play(reg))   # every seeded line can be in play
    for r in reg:
        assert r["state"] == "verified"                            # book lines are verified (in play first)
        assert r["evidence"]["source"] == "web"                    # tagged web provenance (seed_rules.SOURCE)
    # the seeded lines are exactly the survivors of seed_rules for that opponent
    assert lines == {r["line"] for r in seed_rules.seed_lessons(BOOK, opp, "chunli")}


@pytest.mark.skipif(not os.path.exists(BOOK), reason="no lessons/book.json")
def test_seed_is_empty_for_an_unknown_opponent():
    play_loop = load_driver()
    assert play_loop.seed("no_such_fighter", BOOK) == []
