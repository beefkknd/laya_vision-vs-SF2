"""scripts/train.py's opt-in flags for the second run (docs/reviews/2026-09-30_dr_fable_lv_value.md B.1-2):
``--balance sampling`` (check the sampling shares the trainer will really use, not row counts), ``--select nll``
(keep best and early-stop on validation NLL over all rows), the per-character value cross-entropy next to its prior
in train_log.json, and ``--resume-guard`` (on by default: never overwrite a run). No model is loaded here."""
import json
import math
import os
import subprocess
import sys

import pytest

from sf2.data import train_data as TD
from sf2.data import train_select as TS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(REPO, "scripts", "train.py")


def ex(dataset, i, split="t"):
    return {"dataset": dataset, "id": "%s-%d" % (dataset, i),
            "state": {"images": ["frames/%s%d_now.png" % (split, i)]}, "q": {"t": "choice"}}


def rows(counts, split="t"):
    return [ex(d, i, split) for d, n in counts.items() for i in range(n)]


DIRS = ["/x/chunli", "/x/ryu"]


# --- --balance sampling: the sampling-share check ---

def test_sampling_check_passes_unequal_row_counts_that_the_row_check_refuses():
    train, val = rows({"chunli": 900, "ryu": 100}), rows({"chunli": 60, "ryu": 20}, "v")
    assert any("train rows unequal" in p for p in TD.coverage_problems(train, val, DIRS))
    assert not any("rows unequal" in p for p in TD.coverage_problems(train, val, DIRS, share_check=False))
    assert TD.sampling_problems(train, val, DIRS, min_train=100, min_val=20) == []


def test_sampling_check_reads_the_trainers_own_weights(monkeypatch):
    """The check asks laya's mix_probabilities with the arguments train.py passes; a size-weighted mix is refused."""
    train, val = rows({"chunli": 900, "ryu": 100}), rows({"chunli": 60, "ryu": 20}, "v")
    monkeypatch.setattr(TD, "MIX_ALPHA", 1.0)
    probs = TD.sampling_problems(train, val, DIRS, min_train=1, min_val=1)
    assert any("sampling shares unequal" in p for p in probs), probs
    monkeypatch.setattr(TD, "MIX_ALPHA", 0.0)
    monkeypatch.setattr(TD, "MIX_WEIGHTS", {"chunli": 2.0})
    assert any("sampling shares unequal" in p for p in TD.sampling_problems(train, val, DIRS, min_train=1, min_val=1))


def test_sampling_shares_are_equal_under_the_defaults():
    shares = TD.sampling_shares(rows({"chunli": 900, "ryu": 100}))
    assert shares == pytest.approx({"chunli": 0.5, "ryu": 0.5})


def test_sampling_check_needs_rows_in_every_dir():
    train, val = rows({"chunli": 900, "ryu": 99}), rows({"chunli": 60, "ryu": 19}, "v")
    probs = TD.sampling_problems(train, val, DIRS, min_train=100, min_val=20)
    assert any(p.startswith("ryu has 99 train rows < 100") for p in probs), probs
    assert any(p.startswith("ryu has 19 val rows < 20") for p in probs), probs
    none = TD.sampling_problems(rows({"chunli": 900}), val, DIRS, min_train=1, min_val=1)
    assert any("ryu" in p and "not sampled" in p for p in none), none


def test_sampling_check_refuses_a_group_that_is_no_dir():
    train = rows({"chunli": 10, "ryu": 10}) + [dict(ex("chunli", 99), dataset="")]
    probs = TD.sampling_problems(train, rows({"chunli": 5, "ryu": 5}, "v"), DIRS, min_train=1, min_val=1)
    assert any("not a --data dir" in p for p in probs), probs


def test_default_coverage_check_still_compares_row_counts():
    train, val = rows({"chunli": 900, "ryu": 100}), rows({"chunli": 20, "ryu": 20}, "v")
    assert "train rows unequal across datasets: {'chunli': 900, 'ryu': 100}" in TD.coverage_problems(train, val, DIRS)


def test_training_passes_the_checked_sampling_arguments():
    """train.py hands vt.train exactly the arguments the check read (no silent divergence)."""
    src = open(TRAIN).read()
    for arg in ("balance_key=TD.BALANCE_KEY", "mix_weights=TD.MIX_WEIGHTS", "mix_alpha=TD.MIX_ALPHA"):
        assert arg in src, arg


# --- --select nll and the value cross-entropy log ---

def metrics(acc, nll, per=None):
    m = {"all": {"n": 10, "acc": acc, "ece": 0.0, "nll": nll}}
    for name, (x, px) in (per or {}).items():
        m[name] = {"n": 5, "acc": acc, "ece": 0.0, "nll": nll, "n_score": 3, "mae": 0.1, "xent": x, "prior_xent": px}
    return m


def run(seq, select, patience):
    saved = []
    sel = TS.Selection(lambda step: seq[step // 10], lambda: saved.append(True), select, patience)
    stopped = None
    for step in range(0, 10 * len(seq), 10):
        try:
            sel(step)
        except TS.EarlyStop:
            stopped = step
            break
    return sel, saved, stopped


def test_select_acc_is_the_first_runs_rule():
    # the first run's failure: accuracy flat while NLL falls -> stops after ``patience`` evals without gain
    seq = [metrics(0.741, 0.84), metrics(0.742, 0.80), metrics(0.742, 0.78), metrics(0.741, 0.76), metrics(0.742, 0.74)]
    sel, saved, stopped = run(seq, "acc", 3)
    assert sel.best == {"acc": 0.742, "step": 10, "bad": 3} and stopped == 40 and len(saved) == 2


def test_select_nll_keeps_the_lowest_nll_and_does_not_stop_while_it_falls():
    seq = [metrics(0.741, 0.84), metrics(0.742, 0.80), metrics(0.742, 0.78), metrics(0.741, 0.76), metrics(0.742, 0.74)]
    sel, saved, stopped = run(seq, "nll", 3)
    assert stopped is None and sel.best == {"nll": 0.74, "step": 40, "bad": 0} and len(saved) == 5


def test_select_nll_stops_after_patience_evals_without_a_lower_nll():
    seq = [metrics(0.5, 0.80), metrics(0.9, 0.81), metrics(0.9, 0.80), metrics(0.9, 0.82)]
    sel, saved, stopped = run(seq, "nll", 3)
    assert stopped == 30 and sel.best["step"] == 0 and len(saved) == 1


def test_start_best_acc_is_unchanged_and_nll_starts_at_infinity():
    assert TS.start_best("acc") == {"acc": -1.0, "step": None, "bad": 0}
    assert TS.start_best("nll") == {"nll": math.inf, "step": None, "bad": 0}
    with pytest.raises(ValueError):
        TS.start_best("loss")


def test_value_xent_per_character_next_to_its_prior():
    m = metrics(0.7, 0.8, {"chunli": (0.724, 0.748), "guile": (0.857, 0.863)})
    assert TS.value_xent(m) == {"chunli": {"n_score": 3, "xent": 0.724, "prior_xent": 0.748},
                                "guile": {"n_score": 3, "xent": 0.857, "prior_xent": 0.863}}
    assert TS.value_xent(metrics(0.7, 0.8)) == {}                      # no score rows: nothing to log
    no_prior = metrics(0.7, 0.8, {"ryu": (1.0, 1.1)})
    del no_prior["ryu"]["prior_xent"]
    assert TS.value_xent(no_prior) == {"ryu": {"n_score": 3, "xent": 1.0, "prior_xent": None}}


def test_hist_entries_carry_the_value_xent_and_the_selected_metric():
    seq = [metrics(0.7, 0.8, {"chunli": (0.72, 0.75)})]
    sel, _, _ = run(seq, "nll", 3)
    h = sel.hist[0]
    assert h["step"] == 0 and h["all"]["nll"] == 0.8
    assert h["value_xent"] == {"chunli": {"n_score": 3, "xent": 0.72, "prior_xent": 0.75}}


def test_the_selection_does_not_mutate_the_metrics():
    m = metrics(0.7, 0.8, {"chunli": (0.72, 0.75)})
    before = json.dumps(m, sort_keys=True)
    run([m], "nll", 3)
    assert json.dumps(m, sort_keys=True) == before


# --- the defaults and --resume-guard ---

def train_script():
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    try:
        import train as T
    finally:
        sys.path.pop(0)
    return T


def parse(*extra):
    return train_script().parse_args(["--data", "d", "--out", "o"] + list(extra))


def test_train_py_wires_select_and_patience_into_its_eval_fn():
    seq = [metrics(0.9, 0.80), metrics(0.8, 0.70), metrics(0.7, 0.60)]
    for flags, best_step in (((), 0), (("--select", "nll"), 20)):
        a = parse("--patience", "5", *flags)
        sel = train_script().selection(a, lambda step: seq[step // 10], lambda: None)
        for step in (0, 10, 20):
            sel(step)
        assert sel.best["step"] == best_step and sel.patience == 5


def test_train_py_data_gate_by_balance():
    T = train_script()
    train, val = rows({"chunli": 1500, "ryu": 1000}), rows({"chunli": 150, "ryu": 100}, "v")
    args = parse()
    args.data = DIRS
    assert any("train rows unequal" in p for p in T.data_problems(train, val, args))
    args.balance = "sampling"
    assert not [p for p in T.data_problems(train, val, args) if "unequal" in p or " has " in p]


def test_defaults_are_the_first_runs_recipe():
    a = parse()
    assert (a.select, a.balance, a.eval_every, a.patience, a.resume_guard) == ("acc", "rows", 250, 3, True)
    assert (a.rank, a.alpha, a.epochs, a.batch_size, a.lr_head, a.lr_backbone, a.seed) == (16, 32.0, 2.0, 8, 1e-4,
                                                                                         2e-4, 0)


def test_flags_parse():
    a = parse("--select", "nll", "--balance", "sampling", "--eval-every", "1000", "--patience", "5",
              "--no-resume-guard")
    assert (a.select, a.balance, a.eval_every, a.patience, a.resume_guard) == ("nll", "sampling", 1000, 5, False)
    assert "select" in vars(a)                       # train_log.json records vars(args)


def test_resume_guard_refuses_an_existing_out(tmp_path):
    out = tmp_path / "lv_value"
    out.mkdir()
    (out / "train_log.json").write_text("{}")
    p = subprocess.run([sys.executable, TRAIN, "--data", str(tmp_path / "no_such_dir"), "--out", str(out)],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "exists" in p.stderr and "--no-resume-guard" in p.stderr, p.stderr
    assert (out / "train_log.json").read_text() == "{}"


def test_resume_guard_problem_is_pure(tmp_path):
    assert TS.out_problem(str(tmp_path / "new"), True) is None
    assert "exists" in TS.out_problem(str(tmp_path), True)
    assert TS.out_problem(str(tmp_path), False) is None
