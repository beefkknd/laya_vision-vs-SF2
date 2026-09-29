"""The game's words, in one place (sf2.vocab). The values are part of what laya-vision and text laya were trained
on: changing one silently changes their inputs, so they are pinned here."""
import ast
import glob

from sf2 import vocab


def test_values_the_models_were_trained_on():
    assert vocab.FIGHTERS == ("ryu", "honda", "blanka", "guile", "ken", "chunli", "zangief", "dhalsim")
    assert vocab.IDS["chunli"] == 5 and vocab.CHARACTERS[11] == "vega"
    assert (vocab.CLOSE, vocab.MID) == (55, 120)
    assert [vocab.range_of(g) for g in (54, 55, 119, 120)] == ["close", "mid", "mid", "far"]
    assert vocab.RANGE_WORDS == {"close": "up close", "mid": "at mid range", "far": "far away"}
    assert [vocab.bar(v) for v in (176, 159, 158, 106, 53, 52, 250)] == ["full", "full", "high", "high", "half", "low",
                                                                         "low"]
    assert vocab.BARS == ("full", "high", "half", "low")
    assert vocab.OPP_STATES == ("jumping", "crouching", "attacking", "standing", "stunned")


NAMES = {"RANGES", "RANGE_WORDS", "OPP_STATES", "BARS", "FULL_LIFE", "CLOSE", "MID", "IDS", "CHARACTERS",
         "FIGHTERS", "bar", "range_of"}


def test_the_words_are_defined_only_in_vocab():              # invariant: no second copy anywhere
    dup = []
    for f in glob.glob("sf2/**/*.py", recursive=True) + glob.glob("scripts/*.py"):
        if f == "sf2/vocab.py":
            continue
        for n in ast.parse(open(f).read()).body:
            names = [t.id for t in getattr(n, "targets", []) if isinstance(t, ast.Name)]
            names += [n.name] if isinstance(n, (ast.FunctionDef, ast.ClassDef)) else []
            dup += ["%s: %s" % (f, x) for x in names if x in NAMES]
    assert dup == []
