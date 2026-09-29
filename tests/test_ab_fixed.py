"""ab_memory's fixed arms: an arm that plays with lines given on the command line (e.g. a defense lesson to test)."""
import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ab():
    if os.path.join(HERE, "scripts") not in sys.path:
        sys.path.insert(0, os.path.join(HERE, "scripts"))
    spec = importlib.util.spec_from_file_location("ab_memory", os.path.join(HERE, "scripts", "ab_memory.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fixed_arms_parse_and_become_frozen_memories():
    m = ab()
    fixed = m.parse_fixed(["hard=always block_low when he attacks", "two=avoid sweep up close; use more mp at mid range"])
    assert fixed == {"hard": ["always block_low when he attacks"],
                     "two": ["avoid sweep up close", "use more mp at mid range"]}
    mem = m.arm_memory("chunli", "ken", "two", [], fixed)
    assert [x["text"] for x in mem["lessons"]] == ["avoid sweep up close", "use more mp at mid range"]


@pytest.mark.parametrize("bad", ["noequals", "=line", "none=avoid sweep", "x="])
def test_bad_fixed_arms_are_refused(bad):
    with pytest.raises(SystemExit):
        ab().parse_fixed([bad])
