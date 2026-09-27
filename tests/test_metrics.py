"""Training-progress metrics: leak-free validation and per-situation breakdowns."""
import math

from sf2 import actions as A
from sf2 import metrics as M


def _rec(i, episode, rnd, frame, label, text="me=chunli opp=dhalsim dist=mid my_hp=100 opp_hp=100 last=idle "
                                            "airborne=0 opp_airborne=0", my_hp=176, opp_hp=176, hot=False,
         dataset="d"):
    return {"id": "r%d" % i, "episode": episode, "label": label, "state_text": text, "dataset": dataset,
            "meta": {"episode": episode, "round": rnd, "frame": frame, "my_hp": my_hp, "opp_hp": opp_hp, "hot": hot}}


def test_holdout_keeps_whole_rounds_on_one_side():
    recs = [_rec(i, e, r, 100 * i, 0) for i, (e, r) in enumerate((e, r) for e in range(6) for r in range(3)
                                                                   for _ in range(20))]
    val_ids = M.holdout_round_ids(recs, every=4)
    assert val_ids  # something is held out
    by_round = {}
    for rec in recs:
        by_round.setdefault((rec["dataset"], rec["episode"], rec["meta"]["round"]), set()).add(rec["id"] in val_ids)
    assert all(len(sides) == 1 for sides in by_round.values())  # no round is split between train and val


def test_holdout_is_stable_across_calls_and_datasets_do_not_collide():
    a = [_rec(i, 0, i % 5, i, 0, dataset="a") for i in range(50)]
    b = [_rec(100 + i, 0, i % 5, i, 0, dataset="b") for i in range(50)]
    assert M.holdout_round_ids(a + b, every=3) == M.holdout_round_ids(a + b, every=3)


def test_time_into_round_counts_from_the_rounds_first_frame():
    recs = [_rec(0, 0, 1, 2540, 0), _rec(1, 0, 1, 3140, 0), _rec(2, 0, 0, 0, 0)]
    M.annotate(recs)
    assert [r["meta"]["t_round"] for r in recs] == [0.0, 10.0, 0.0]  # seconds at 60 fps


def _row(label, probs, t=0.0, **kw):
    rec = _rec(0, 0, 0, 0, label, **kw)
    rec["meta"]["t_round"] = t
    return rec, [math.log(max(p, 1e-9)) for p in probs], [1.0 if j == label else 0.0 for j in range(len(probs))]


def _probs(pick, k=len(A.ACTIONS), p=0.9):
    return [p if j == pick else (1 - p) / (k - 1) for j in range(k)]


def test_by_move_accuracy_separates_kicks_from_forward():
    fw, hk = A.ACTIONS.index("forward"), A.ACTIONS.index("hk")
    rows = [_row(fw, _probs(fw)), _row(fw, _probs(fw)), _row(hk, _probs(fw)), _row(hk, _probs(hk))]
    s = M.slice_metrics([r for r, _, _ in rows], [z for _, z, _ in rows], [t for _, _, t in rows])
    assert s["by_move"]["forward"]["acc"] == 1.0
    assert s["by_move"]["hk"]["acc"] == 0.5
    assert s["by_move"]["hk"]["n"] == 2
    assert s["confusion"]["hk"]["forward"] == 1  # label said hk, model said forward
    assert abs(s["all"]["acc"] - 0.75) < 1e-9


def test_slices_cover_phase_hp_distance_and_danger():
    fw = A.ACTIONS.index("forward")
    rows = [
        _row(fw, _probs(fw), t=3.0, text="me=chunli stand hp=100 opp=dhalsim jump hp=100 dist=close facing=right "
                                           "corner=none time=early last=idle"),
        _row(fw, _probs(0), t=40.0, my_hp=30, opp_hp=150, hot=True,
             text="me=chunli jump hp=17 opp=dhalsim attack hp=85 dist=far facing=left corner=me time=mid last=hk"),
    ]
    s = M.slice_metrics([r for r, _, _ in rows], [z for _, z, _ in rows], [t for _, _, t in rows])
    assert s["by_time"]["early"]["acc"] == 1.0 and s["by_time"]["late"]["acc"] == 0.0
    assert s["by_hp"]["endgame"]["n"] == 1 and s["by_hp"]["opening"]["n"] == 1
    assert set(s["by_dist"]) == {"close", "far"}
    assert s["by_opp_air"]["air"]["n"] == 1 and s["by_opp_air"]["ground"]["n"] == 1
    old_note = "me=chunli opp=dhalsim dist=close my_hp=100 opp_hp=100 last=idle airborne=0 opp_airborne=1"
    assert M.situation(_rec(0, 0, 0, 0, 0, text=old_note))["by_opp_air"] == "air"   # the note before 2026-09-26
    kick = "me=chunli stand hp=100 opp=dhalsim jumpattack hp=100 dist=close facing=right corner=none time=early"
    assert M.situation(_rec(0, 0, 0, 0, 0, text=kick))["by_opp_air"] == "air"
    assert s["by_danger"]["hit_next"]["acc"] == 0.0
    assert 0.0 < s["all"]["p_label"] < 1.0


def test_summary_line_names_moves_and_phases():
    fw = A.ACTIONS.index("forward")
    rows = [_row(fw, _probs(fw), t=3.0)]
    line = M.summary(M.slice_metrics([r for r, _, _ in rows], [z for _, z, _ in rows], [t for _, _, t in rows]))
    assert "forward" in line and "early" in line


def test_label_probability_of_the_models_choice_reads_the_soft_target():
    # the label splits 0.6 forward / 0.4 hk; a model that picks hk is 0.4 right, not 0 (accuracy) or 1
    fw, hk = A.ACTIONS.index("forward"), A.ACTIONS.index("hk")
    rec, z, _ = _row(fw, _probs(hk))
    soft = [0.6 if j == fw else 0.4 if j == hk else 0.0 for j in range(len(A.ACTIONS))]
    s = M.slice_metrics([rec], [z], [soft])
    assert abs(s["all"]["t_of_pred"] - 0.4) < 1e-9
    assert s["all"]["acc"] == 0.0
