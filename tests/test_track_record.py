"""The track record (sf2.system2.track_record): per opponent and lesson line, the rounds it was in play against the same
round without advice, over earlier runs (the run is the unit). Round 3, 2026-09-29: "use more forward at mid range when
he jumps" passed the per-situation check in every Honda run while its rounds lost 16 hp to no advice; only the round
outcome, pooled across runs, shows it. Qwen sees the record; the verifier refuses a lesson that clearly hurts."""
import importlib.util
import inspect
import json
import os
import sys

from sf2.system2 import lessons as L
from sf2.system2 import track_record as T

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAD = "use more forward at mid range when he jumps"
GOOD = "use more hp up close when he jumps"


def rnd(hp, lines):
    return {"dealt": max(hp, 0), "taken": max(-hp, 0), "lines": lines, "result": "win" if hp > 0 else "loss"}


def a_run(root, name, loop, none, prompt="character"):
    d = os.path.join(root, name)
    for arm, rs in (("loop", loop), ("none", none)):
        os.makedirs(os.path.join(d, arm))
        with open(os.path.join(d, arm, "rounds.jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rs))
    with open(os.path.join(d, "verdict.json"), "w") as f:
        json.dump({"seed": 1, "prompt": prompt}, f)
    return d


def world(root, runs=4):
    """In each run: BAD in play for 6 rounds losing ~20 to no advice, GOOD for 6 rounds winning ~15, 3 plain rounds."""
    for i in range(runs):
        loop = [rnd(-20 + (i + k) % 5, [BAD]) for k in range(6)] + [rnd(15 + (i * k) % 4, [GOOD]) for k in range(6)] + \
               [rnd(0, []) for _ in range(3)]
        a_run(root, "2026092%d-000000_honda" % i, loop, [rnd(0, []) for _ in loop])


def test_a_lesson_whose_rounds_lose_to_no_advice_hurts(tmp_path):
    world(str(tmp_path))
    t = T.build([str(tmp_path)])
    rec = t["opponents"]["honda"]
    assert rec[BAD]["verdict"] == "hurts" and rec[BAD]["runs"] == 4 and rec[BAD]["rounds"] == 24
    assert rec[GOOD]["verdict"] == "helps" and rec[BAD]["hi"] < 0 < rec[GOOD]["lo"]


def test_one_run_is_too_few_however_many_rounds(tmp_path):
    loop = [rnd(-20 - k % 3, [BAD]) for k in range(30)]
    a_run(str(tmp_path), "20260929-000000_honda", loop, [rnd(0, [])] * 30)
    rec = T.build([str(tmp_path)])["opponents"]["honda"][BAD]
    assert rec["rounds"] == 30 and rec["verdict"] == "few"


def test_unfinished_runs_and_unequal_arms_are_left_out_and_listed(tmp_path):
    world(str(tmp_path), runs=3)
    a_run(str(tmp_path), "20260929-000009_honda", [rnd(-50, [BAD])] * 10, [rnd(0, [])] * 9)       # unequal
    os.makedirs(os.path.join(str(tmp_path), "20260929-000010_honda", "loop"))                   # no verdict
    t = T.build([str(tmp_path)])
    assert t["opponents"]["honda"][BAD]["runs"] == 3
    assert [os.path.basename(s) for s in t["skipped"]] == ["20260929-000009_honda"]


def test_the_build_is_deterministic(tmp_path):
    world(str(tmp_path))
    assert json.dumps(T.build([str(tmp_path)])) == json.dumps(T.build([str(tmp_path)]))


def test_the_verifier_does_not_refuse_on_the_record(tmp_path):
    """Review 2026-09-29 (finding 4): refusing on the record switched the loop off against Honda, and the blamed lesson
    was never in play alone. The record is shown to Qwen; the verifier ignores it."""
    world(str(tmp_path))
    track = T.build([str(tmp_path)])["opponents"]["honda"]
    assert track[BAD]["verdict"] == "hurts"
    claim = {"kind": "use_more", "move": "forward", "range": "mid", "when": "jumping"}
    assert "track" not in inspect.signature(L.propose).parameters
    _, out = L.propose([], [claim], [], 0, moves=["forward"])
    assert out[0]["state"] == "testing"
    assert not hasattr(T, "violations") and not hasattr(T, "hurts")          # nothing left that could refuse


def test_the_prompt_shows_the_record_worst_first(tmp_path):
    world(str(tmp_path))
    track = dict(T.build([str(tmp_path)])["opponents"]["honda"],
                 **{"always block_low at mid range when he attacks": T.record([[1, -1, 2]] * 5),        # unclear
                    "avoid sweep up close": T.record([[30, 31, 29]] * 5)})                              # helps
    text = T.prompt_lines(track)
    assert text[0].startswith("- " + BAD) and "hurts" in text[0] and "4 runs" in text[0]
    assert [x.split(": ")[-1].split(" - ")[0] for x in text] == ["hurts", "helps", "helps", "unclear"]
    assert text[1].startswith("- " + GOOD)                                  # helps: the lower mean first


def test_prompts_without_a_track_are_unchanged_and_with_one_show_it(tmp_path):
    from sf2.system2 import character_prompt as C, lesson_prompt as P
    world(str(tmp_path))
    track = T.build([str(tmp_path)])["opponents"]["honda"]
    rows = [{"action": "sweep", "range": "mid", "kind": "attack", "dealt": 3, "taken": 0, "opp_air": False,
             "opp_state": "stand", "opp_reaction": ["stand"], "actual": "hit"}] * 12
    for mod in (C, P):
        plain = mod.messages("chunli", "honda", [], rows, rows, [], [], ["sweep"])
        assert plain == mod.messages("chunli", "honda", [], rows, rows, [], [], ["sweep"], track=None)
        user = mod.messages("chunli", "honda", [], rows, rows, [], [], ["sweep"], track=track)[1]["content"]
        assert "TRACK RECORD" in user and BAD in user and "TRACK RECORD" not in plain[1]["content"]


def qwen_lessons():
    sys.path.insert(0, os.path.join(HERE, "scripts"))
    spec = importlib.util.spec_from_file_location("qwen_lessons", os.path.join(HERE, "scripts", "qwen_lessons.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_loop_reads_a_track_file_for_its_opponent(tmp_path):
    world(str(tmp_path))
    path = os.path.join(str(tmp_path), "track.json")
    with open(path, "w") as f:
        json.dump(T.build([str(tmp_path)]), f)
    q = qwen_lessons()
    track, digest = q.load_track(path, "honda")
    assert track[BAD]["verdict"] == "hurts" and len(digest) == 64
    assert q.load_track(path, "ken") == ({}, digest)
    assert q.load_track(None, "honda") == (None, None)


def test_qwen_copying_the_words_of_a_lesson_line_still_parses():
    """Round 4: 16 claims came back with "when": "jumps" (the track record shows lines in text laya's words)."""
    from sf2.system2 import character_prompt as C, lesson_prompt as P
    for mod, view in ((C, "answer"), (P, "attack")):
        for verb, when in (("jumps", "jumping"), ("he attacks", "attacking"), ("when he crouches", "crouching"),
                           ("stands", "standing"), ("is stunned", "stunned")):
            claims, problems = mod.parse_claims({view: {"kind": "use_more", "move": "c.mk", "range": "mid",
                                                        "when": verb}})
            assert claims[0]["when"] == when and not problems


def test_the_record_does_not_forbid_anything(tmp_path):
    world(str(tmp_path))
    part = T.prompt_part("honda", T.build([str(tmp_path)])["opponents"]["honda"])
    assert "do not propose" not in part and "refused" not in part and BAD in part


def crowd(root, runs=4):
    """Each run: BAD in play 6 rounds with X, 4 of them also with Y, 2 with Z, 1 with W; GOOD 6 rounds alone."""
    for i in range(runs):
        loop = ([rnd(-20 + (i + k) % 5, [BAD, X] + ([Y] if k < 4 else []) + ([Z] if k < 2 else [])
                     + ([W] if k < 1 else [])) for k in range(6)]
                + [rnd(15 + (i * k) % 4, [GOOD]) for k in range(6)])
        a_run(root, "2026092%d-000000_honda" % i, loop, [rnd(0, []) for _ in loop])


X, Y, Z, W = ("avoid sweep at mid range when he jumps", "always block_low at mid range when he attacks",
              "avoid c.mk up close", "use more throw up close")


def test_the_build_records_the_lessons_most_often_in_play_with_each_line(tmp_path):
    crowd(str(tmp_path))
    rec = T.build([str(tmp_path)])["opponents"]["honda"]
    assert rec[BAD]["with"] == [[X, 24], [Y, 16], [Z, 8]]                    # top 3 by rounds together, W left out
    assert rec[X]["with"][0] == [BAD, 24] and rec[GOOD]["with"] == []          # GOOD was always in play alone
    assert rec[Z]["with"] == [[Y, 8], [X, 8], [BAD, 8]]                       # ties: in name order


def test_the_prompt_shows_what_each_line_was_in_play_with(tmp_path):
    crowd(str(tmp_path))
    text = T.prompt_lines(T.build([str(tmp_path)])["opponents"]["honda"])
    bad = [x for x in text if x.startswith("- " + BAD + ":")][0]
    assert bad.endswith("(in play with: %s, %s, %s)" % (X, Y, Z))
    assert [x for x in text if x.startswith("- " + GOOD + ":")][0].endswith("helps")       # alone: nothing added


def test_the_build_with_co_lines_is_deterministic(tmp_path):
    crowd(str(tmp_path))
    assert json.dumps(T.build([str(tmp_path)])) == json.dumps(T.build([str(tmp_path)]))


def test_a_record_file_without_co_lines_still_shows(tmp_path):
    """Files built before the co-lines were recorded (lessons/track_record.json at 235d470) still read."""
    r = T.record([[-20, -21, -19, -22]] * 4)
    assert T.prompt_lines({BAD: r})[0].endswith("hurts")


def test_a_short_smoke_run_is_not_evidence(tmp_path):
    """Harness ledger 2026-09-30: two 2-round smoke runs (20260929-134810_ken, -145756_ken) were counted as runs."""
    world(str(tmp_path), runs=3)
    a_run(str(tmp_path), "20260929-000009_honda", [rnd(-90, [BAD])] * 2, [rnd(0, [])] * 2)
    t = T.build([str(tmp_path)])
    assert t["opponents"]["honda"][BAD]["runs"] == 3
    assert [os.path.basename(s) for s in t["short"]] == ["20260929-000009_honda"]
