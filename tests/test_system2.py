"""System 2's short memory and playbook: the prompt it reads, what it may keep (vet, fits_laya), and the reply checks."""


def test_short_memory_goes_into_the_prompt_and_bad_memory_is_refused():
    from sf2.system2.memory import check, prompt_text
    note = "me=ryu opp=ken dist=mid side=left dx=+70"
    assert prompt_text(note, None) == note and prompt_text(note, {"opp": "ken", "lessons": []}) == note
    les = [{"text": "use more sweep at mid range", "kind": "use_more", "action": "sweep", "range": "mid",
            "evidence": {"tries": 11, "count": 7, "rate": 0.64, "refs": ["g1f40"]}}]
    assert prompt_text(note, {"opp": "ken", "lessons": les * 7}).count("use more sweep") == 5   # at most 5 lessons
    assert prompt_text(note, {"opp": "ken", "lessons": les}) == note + "\nmemory vs ken: use more sweep at mid range"
    acts = ["lp", "sweep", "hadoken"]
    assert check({"lessons": les}, acts) == []
    bad = [dict(les[0], action="flying_kick"), dict(les[0], kind="vibes"),
           dict(les[0], evidence={"tries": 3, "count": 5, "refs": ["g1f1"]}), dict(les[0], evidence={})]
    assert all(check({"lessons": [b]}, acts) for b in bad)

def test_system2_vet_keeps_laya_lessons_short_unique_and_backed():
    from sf2.system2.system2 import vet
    def act(action, rng, actual, hit_me=False, opp_state="stand"):
        return {"kind": "attack", "action": action, "range": rng, "actual": actual, "i_was_hit": hit_me,
                "opp_state": opp_state, "game": 0, "frame": 0}
    acts = [act("spinning_bird_kick", r, "whiff", True) for r in ("close", "mid", "far") for _ in range(4)]
    acts += [act("lp", "close", "hit") for _ in range(5)]
    L = lambda text, kind, action, rng, claim: dict(text=text, kind=kind, action=action, range=rng, claim=claim)  # noqa
    kept, rej = vet([L("avoid spinning_bird_kick: he punishes it", "avoid", "spinning_bird_kick", None, "punished"),
                     L("avoid spinning_bird_kick at mid", "avoid", "spinning_bird_kick", "mid", "punished"),   # repeat
                     L("use more lp up close: lands 57%", "use_more", "lp", "close", "lands"),                 # number
                     L("use more lp up close, it is by far your most dependable poke there", "use_more", "lp",
                       "close", "lands"),                                                                    # too long
                     L("use more lp up close", "use_more", "lp", "close", "lands"),
                     L("use more mp at mid", "use_more", "mp", "mid", "lands")], acts, 5)                     # no tries
    assert [k["text"] for k in kept] == ["avoid spinning_bird_kick: he punishes it", "use more lp up close"]
    assert kept[0]["evidence"]["tries"] == 12 and kept[0]["evidence"]["count"] == 12
    whys = [r["why"] for r in rej]
    assert any("repeats" in w for w in whys) and any("numbers" in w for w in whys)
    assert any("characters" in w for w in whys) and any("tries" in w for w in whys)

def test_short_memory_evidence_uses_all_games_until_this_opponent_has_enough():
    from sf2.system2.system2 import vet
    a = lambda act, res, st="stand": {"kind": "attack", "action": act, "range": "close", "actual": res,  # noqa: E731
                                       "i_was_hit": False, "opp_state": st, "game": 0, "frame": 0}
    all_games = [a("lp", "hit") for _ in range(20)]
    vs_him = [a("lp", "whiff"), a("hp", "hit", "jump"), a("hp", "hit", "jump"), a("hp", "hit", "jump")]
    L = lambda text, kind, action, claim: dict(text=text, kind=kind, action=action, range=None, claim=claim)  # noqa
    kept, rej = vet([L("use more lp up close", "use_more", "lp", "lands"),
                     L("when he jumps in, use hp", "counter", "hp", "counter:jump"),
                     L("he jumps in a lot", "opponent_habit", None, "habit:jump")], vs_him, 5, fallback=all_games)
    by = {k["text"]: k["evidence"] for k in kept}
    assert by["use more lp up close"]["scope"] == "all opponents" and by["use more lp up close"]["tries"] == 20
    assert by["when he jumps in, use hp"]["scope"] == "this opponent" and by["when he jumps in, use hp"]["tries"] == 3
    assert by["he jumps in a lot"]["tries"] == 4                          # habits: his games only, never the fallback

def test_a_kept_short_memory_must_meet_todays_rules():
    from sf2.system2.system2 import fits_laya
    ok = {"text": "use more lp up close", "kind": "use_more", "action": "lp", "range": "close", "claim": "lands"}
    old_style = {"text": "ryu punishes my spinning_bird_kick at mid range (70%)", "kind": "avoid",
                 "action": "spinning_bird_kick", "range": "mid"}                    # no claim, numbers: the old builder
    assert fits_laya({"lessons": [ok]})
    assert not fits_laya({"lessons": [ok, old_style]})
    assert not fits_laya({"lessons": [ok, dict(ok, text="use more lp up close, really")]})        # the same move twice
    assert not fits_laya({"lessons": [dict(ok, text="x" * 61)]}) and not fits_laya({"lessons": []}) and not fits_laya(None)

def test_revise_prompt_shows_which_lessons_system1_followed():
    from sf2.system2.prompts import followed
    a = lambda act, rng, res, punished=False: {"action": act, "range": rng, "actual": res, "i_was_hit": punished}  # noqa
    recent = [a("sweep", "close", "whiff", True), a("sweep", "close", "whiff"), a("hp", "mid", "hit")]
    mem = {"lessons": [{"text": "use more lp up close", "kind": "use_more", "action": "lp", "range": "close"},
                       {"text": "avoid sweep up close", "kind": "avoid", "action": "sweep", "range": "close"},
                       {"text": "he jumps in at far", "kind": "opponent_habit", "action": None, "range": "far"}]}
    lines = followed(recent, mem)
    assert lines[0] == '- IGNORED (never used) "use more lp up close": lp at close used 0 times'
    assert lines[1] == '- IGNORED (used anyway) "avoid sweep up close": sweep at close used 2 times (whiff 2; punished 1)'
    assert len(lines) == 2                                          # habits have no move to follow

def test_system2_reply_checks_flag_each_broken_rule():
    """The checks the live Qwen prompt tests gate on, seen red on canned bad replies."""
    import json
    from sf2.system2.checks import reply_problems
    logs = json.load(open("tests/fixtures/system2/chunli_logs.json"))
    ryu = logs["ryu"]["actions"]
    every = [a for v in logs.values() for a in v["actions"]]
    good = {"text": "use more lp up close", "kind": "use_more", "action": "lp", "range": "close", "claim": "lands"}
    assert reply_problems("chunli", {"lessons": [good]}, 5, ryu, every) == []
    bad = {
        "no lessons key": {"notes": "x"},
        "over the limit": {"lessons": [good, dict(good, action="c.lk", text="use more c.lk up close")] * 3},
        "not her move": {"lessons": [dict(good, action="hadoken", text="use more hadoken")]},
        "numbers": {"lessons": [dict(good, text="use more lp up close: 57%")]},
        "too long": {"lessons": [dict(good, text="use more lp up close because it is by far the best poke you have")]},
        "repeat": {"lessons": [good, dict(good, text="lp up close again")]},
        "use and avoid": {"lessons": [good, dict(good, kind="avoid", claim="whiffs", text="avoid lp up close")]},
        "not backed": {"lessons": [dict(good, action="hp", text="use more hp up close")]},
    }
    for name, reply in bad.items():
        assert reply_problems("chunli", reply, 5, ryu, every), name
