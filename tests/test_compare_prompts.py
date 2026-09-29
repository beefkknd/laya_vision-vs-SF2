"""scripts/compare_prompts.py: lesson-loop runs paired by (opponent, seed), one per prompt, against the same no-advice
arm (byte-identical across prompts, or the pair is refused)."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cp():
    sys.path.insert(0, os.path.join(HERE, "scripts"))
    spec = importlib.util.spec_from_file_location("compare_prompts", os.path.join(HERE, "scripts", "compare_prompts.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(root, name, opp, seed, prompt, loop_hp, none_hp, none_bytes=b"same"):
    d = os.path.join(root, name)
    for arm, hps in (("loop", loop_hp), ("none", none_hp)):
        os.makedirs(os.path.join(d, arm))
        with open(os.path.join(d, arm, "rounds.jsonl"), "w") as f:
            f.write("".join(json.dumps({"dealt": max(h, 0), "taken": max(-h, 0), "result": "win" if h > 0 else "loss"})
                            + "\n" for h in hps))
    with open(os.path.join(d, "none", "actions.jsonl"), "wb") as f:
        f.write(none_bytes)
    v = {"seed": seed, "violations": 0, "failed_jobs": [], "qwen_hold_rate": 0.5, "random_hold_rate": 0.1}
    if prompt:
        v["prompt"] = prompt
    with open(os.path.join(d, "verdict.json"), "w") as f:
        json.dump(v, f)
    return d


def test_runs_pair_by_opponent_and_seed(tmp_path):
    r = str(tmp_path)
    run(r, "1_ken", "ken", 1, None, [10] * 10, [0] * 10)                   # an old run: no prompt in the verdict = views
    run(r, "2_ken_character", "ken", 1, "character", [30] * 10, [0] * 10)
    run(r, "3_ken", "ken", 2, "views", [5] * 10, [0] * 10)                  # no character twin: not paired
    pairs, problems = cp().pairs([r])
    assert list(pairs) == [("ken", 1)] and not problems
    assert cp().diffs(pairs)["b_minus_a"]["ken"] == [20] * 10


def test_a_pair_whose_no_advice_arms_differ_is_refused(tmp_path):
    r = str(tmp_path)
    run(r, "1_ken", "ken", 1, "views", [10] * 10, [0] * 10)
    run(r, "2_ken_character", "ken", 1, "character", [30] * 10, [0] * 10, none_bytes=b"other")
    pairs, problems = cp().pairs([r])
    assert pairs == {} and "differ" in problems[0]


def test_a_repeated_seed_keeps_the_first_run(tmp_path):
    r = str(tmp_path)
    run(r, "1_ken", "ken", 1, "views", [10] * 10, [0] * 10)
    run(r, "5_ken", "ken", 1, "views", [99] * 10, [0] * 10)                 # a repeat of the same seed
    run(r, "2_ken_character", "ken", 1, "character", [30] * 10, [0] * 10)
    pairs, _ = cp().pairs([r])
    assert pairs[("ken", 1)]["views"].endswith("1_ken")


def test_a_run_without_a_verdict_is_ignored(tmp_path):
    r = str(tmp_path)
    run(r, "1_ken", "ken", 1, "views", [10] * 10, [0] * 10)
    os.remove(os.path.join(r, "1_ken", "verdict.json"))
    assert cp().pairs([r]) == ({}, [])


@pytest.mark.parametrize("name,opp", [("20260929-172936_ken", "ken"), ("20260929-173010_honda_character", "honda")])
def test_the_opponent_comes_from_the_run_name(name, opp):
    assert cp().opponent(name) == opp


def test_any_two_prompts_can_be_compared(tmp_path):
    r = str(tmp_path)
    run(r, "1_ken_character", "ken", 1, "character", [10] * 10, [0] * 10)
    run(r, "2_ken_character_fgc", "ken", 1, "character_fgc", [25] * 10, [0] * 10)
    pairs, _ = cp().pairs([r], ("character", "character_fgc"))
    assert cp().diffs(pairs, ("character", "character_fgc"))["b_minus_a"]["ken"] == [15] * 10


def lines(*extra):
    return "".join(json.dumps(dict({"action": "sweep", "taken": 3}, **e)) + "\n" for e in extra).encode()


def test_no_advice_arms_that_differ_only_in_logging_only_fields_still_pair(tmp_path):
    """2026-09-29: two runs re-ran after opp_move / opp_shot were added to the log (logging only, play unchanged)."""
    r = str(tmp_path)
    run(r, "1_ken", "ken", 1, "views", [10] * 10, [0] * 10, none_bytes=lines({"opp_move": "normal", "opp_shot": False}))
    run(r, "2_ken_character", "ken", 1, "character", [30] * 10, [0] * 10, none_bytes=lines({}))
    pairs, problems = cp().pairs([r])
    assert list(pairs) == [("ken", 1)] and not problems


def test_no_advice_arms_that_differ_in_play_are_still_refused(tmp_path):
    r = str(tmp_path)
    run(r, "1_ken", "ken", 1, "views", [10] * 10, [0] * 10, none_bytes=lines({"opp_move": "normal", "taken": 4}))
    run(r, "2_ken_character", "ken", 1, "character", [30] * 10, [0] * 10, none_bytes=lines({}))
    pairs, problems = cp().pairs([r])
    assert pairs == {} and "differ" in problems[0]
