"""Drop rare actions from an act dataset (owner 2026-10-01: "I don't want rare move"; docs/prereg_movement_data.md).

An (actor, code) is rare when the collection saw it in fewer than MIN_EPISODES episodes (the dataset gate's
coverage table, ``episodes_seen``). Its rows go (both its "act" and its "stage" rows), and its option leaves every
"act" question of that actor, with ``label`` re-indexed to the shorter option list. Nothing else changes.
"""
from typing import Dict, Iterable, Iterator, Set, Tuple

MIN_EPISODES = 10
Key = Tuple[str, int]


def rare_codes(table: Dict[str, Dict], min_episodes: int = MIN_EPISODES) -> Set[Key]:
    """table: {"<actor> act<NN>": {"episodes_seen": n, ...}} -> {(actor, NN)} seen in fewer than min_episodes."""
    out = set()
    for name, row in table.items():
        actor, act = name.split()
        if not act.startswith("act"):
            raise ValueError("bad coverage key %r" % name)
        if row["episodes_seen"] < min_episodes:
            out.add((actor, int(act[3:])))
    return out


def _act_name(code: int) -> str:
    return "act%02d" % code


def filter_row(row: Dict, rare: Set[Key]) -> Dict:
    """The row without rare options, or {} when the row's own action is rare. Returns a new dict."""
    if (row["actor"], row["code"]) in rare:
        return {}
    if row["key"] != "act":
        return dict(row)
    drop = {_act_name(c) for a, c in rare if a == row["actor"]}
    criteria = {k: v for k, v in row["question"]["criteria"].items() if k not in drop}
    if row["answer"] not in criteria:
        raise ValueError("%s: answer %s not among its options" % (row["id"], row["answer"]))
    question = {**row["question"], "criteria": criteria}
    return {**row, "question": question, "label": list(criteria).index(row["answer"])}


def filter_rows(rows: Iterable[Dict], rare: Set[Key]) -> Iterator[Dict]:
    for row in rows:
        kept = filter_row(row, rare)
        if kept:
            yield kept
