"""Round facts (code counts), the notebook's checks (change budget, plan lines, experiments) and the prompt."""
from sf2.system2.notebook import changes, check, clean_plan, edits, empty, merged, tries_allowed
from sf2.system2.notebook_prompts import reflect_prompt
from sf2.system2.round_facts import facts, plan_check, text, unexpected

MOVES = ["lp", "mp", "c.mk", "sweep", "spinning_bird_kick", "throw", "block_high", "block_low"]


def act(action, rng="mid", kind="attack", actual="hit", dealt=0, taken=0, hit=False, air=False, attacked=False):
    return {"action": action, "range": rng, "kind": kind, "actual": actual, "dealt": dealt, "taken": taken,
            "i_was_hit": hit, "opp_air": air, "opp_attacked": attacked, "opp_blocked": False,
            "opp_reaction": ["stand"]}


ROUND = [act("sweep", dealt=12), act("sweep", dealt=12), act("sweep", actual="whiff", taken=30, hit=True),
         act("sweep", dealt=12), act("forward", kind="movement", actual="none"),
         act("c.mk", rng="close", actual="whiff", taken=20, hit=True, air=True)]
SUMMARY = {"result": "loss", "dealt": 36, "taken": 50}


def test_facts_count_hit_points_per_move_and_range():
    f = facts(ROUND, SUMMARY)
    assert f["mine"]["sweep@mid"] == {"tries": 4, "hit": 3, "whiff": 1, "blocked": 0, "punished": 1, "hp": 6}
    assert f["hp"] == -14 and f["hit_by"]["jump-in at close while I was attacking"] == 1


def test_unexpected_flags_a_good_move_that_lost():
    f = facts(ROUND + [act("sweep", actual="whiff", taken=40, hit=True)], SUMMARY)
    got = unexpected(f, ["sweep at mid is my best move"], MOVES)
    assert len(got) == 1 and "lost" in got[0]
    assert unexpected(f, ["sweep up close is bad"], MOVES) == []          # wrong range: not contradicted


def test_plan_check_reports_use():
    f = facts(ROUND, SUMMARY)
    assert plan_check(f, ["use more sweep at mid", "try throw up close"], MOVES) == [
        "'use more sweep at mid': used 4 times, +6 hit points", "'try throw up close': not used"]
    assert "nothing" in text(f, [], [], "ryu")


def reply(self_lines, plan, opp_lines=(), questions=()):
    return {"notebook": {"self": list(self_lines), "opponent": list(opp_lines), "questions": list(questions)},
            "plan": list(plan)}


def test_change_budget_trims_instead_of_rejecting():
    old = empty("chunli")
    # an empty book gets the first 4 new lines of an over-long first update (a run where every update was
    # rejected for size left the book empty for 40 rounds)
    first = reply(["a", "b", "c"], ["use more sweep"], opp_lines=["d", "e"], questions=["f"])
    assert check(old, first, "ryu", MOVES, 1)["notebook"] == []
    book = merged(old, first, "ryu")
    assert book["self"] == ["a", "b", "c"] and book["opponents"]["ryu"] == ["d"] and book["questions"] == []
    # a rewrite: at most 2 of the old lines go, at most 4 new come
    book2 = merged(book, reply(["x", "y", "z"], ["use more sweep"], opp_lines=["w"]), "ryu")
    assert sum(changes(book, book2, "ryu")) == 6 and changes(book, book2, "ryu") == (4, 2)
    assert edits(book2, book2, "ryu") == 0


def test_plan_keeps_readable_lines_and_allowed_tries():
    keep, dropped = clean_plan(["use more sweep 3 times", "be brave", "avoid lp up close", "try throw up close",
                                "try lp at mid"], MOVES, 1)
    assert keep == ["avoid lp up close", "try throw up close"] and len(dropped) == 3
    assert clean_plan(["try throw up close", "try lp at mid"], MOVES, 2)[0] == ["try throw up close", "try lp at mid"]


def test_more_experiments_when_losing():
    assert tries_allowed(["win", "win", "win", "win"]) == 0
    assert tries_allowed(["loss", "loss", "loss", "loss"]) == 2
    assert tries_allowed(["win", "loss"]) == 1 and tries_allowed([]) == 1
    assert tries_allowed(["win", "win", "win", "loss"]) == 1           # one loss in four earns an experiment
    assert tries_allowed(["win", "loss", "loss", "loss"]) == 2


def test_prompt_mentions_new_opponent_and_budget():
    p = reflect_prompt("chunli", "ryu", empty("chunli"), "facts", 2)
    assert "ryu is new to you" in p and "2 \"try\" experiments" in p


def test_a_line_with_two_ranges_never_crashes():          # Qwen wrote this (notebook run, 2026-09-28)
    from sf2.system1.advice import read
    from sf2.system1.advisor import shortlist
    line = "avoid spinning_bird_kick at mid and far"
    assert read(line, MOVES).move is None                  # unreadable: ignored, not a crash
    f = facts(ROUND, SUMMARY)
    assert unexpected(f, [line], MOVES) == [] and plan_check(f, [line], MOVES) == []
    assert clean_plan([line], MOVES, 1) == ([], ["names none of my moves, or two ranges at once: %r" % line])
    assert "forward" in shortlist({"lp": 0.4, "sweep": 0.6}, [line], MOVES, ("mid", "standing"))


def test_a_reply_that_is_not_an_object_keeps_book_and_plan():      # valid JSON, wrong shape: never a crash
    from sf2.system2.notebook import apply_reply
    book = merged(empty("chunli"), reply(["a"], ["use more sweep at mid"]), "ryu")
    for bad in ([1, 2], "use more sweep", None, 3):
        nb, plan, why = apply_reply(book, ["avoid lp up close"], bad, "ryu", MOVES, 1)
        assert nb == book and plan == ["avoid lp up close"] and "not a JSON object" in why["error"]
    nb, plan, why = apply_reply(book, [], reply(["a", "b"], ["try throw up close"]), "ryu", MOVES, 1)
    assert nb["self"] == ["a", "b"] and plan == ["try throw up close"] and why["edits"] == 1


def test_trim_edges_duplicates_and_full_sections():
    from sf2.system2.notebook import MAX_LINES, _trim
    # the same line re-worded only in case/punctuation is not new; a repeat in Qwen's list is added once
    assert _trim(["Sweep lands at mid."], ["sweep lands at mid", "throw up close", "throw up close"], [4, 2]) == [
        "Sweep lands at mid.", "throw up close"]
    full = ["line %d" % i for i in range(MAX_LINES)]
    assert _trim(full, full + ["one more"], [4, 2]) == full                 # a full section takes nothing new
    budget = [4, 2]
    assert _trim(full, [], budget) == full[2:] and budget == [4, 0]         # an emptied section loses only 2
