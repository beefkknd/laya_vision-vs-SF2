"""Pure trend-series helpers for the monitor TUI (win-rate climbs, margin sparkline, bar chart)."""
from sf2.eval.tui_model import RoundResult, cum_winrate_series, margin_series, spark, vbars


def test_cum_winrate_climbs():
    assert cum_winrate_series(("W", "L", "W")) == (1.0, 0.5, 2 / 3)
    assert cum_winrate_series(()) == ()
    # a run that learns to win: the win-rate rises from 0 toward a high final value
    s = cum_winrate_series(("L", "L", "W", "W", "W", "W"))
    assert s[0] == 0.0 and s[-1] == 4 / 6 and s[-1] > s[2]


def test_margin_series_per_game():
    rr = [RoundResult(0, 0, "win", 80, 120, 40), RoundResult(1, 0, "loss", -50, 30, 80)]
    assert margin_series(rr) == (80, -50)


def test_spark_monotonic_and_empty():
    assert spark([]) == ""
    s = spark([0, 1, 2, 3, 4, 5, 6, 7])
    assert s[0] == "▁" and s[-1] == "█"
    assert len(s) == 8


def test_vbars_shape_and_rising():
    rows = vbars([0.0, 0.5, 1.0], height=4, lo=0.0, hi=1.0)
    assert len(rows) == 4 and all(len(r) == 3 for r in rows)
    # the 1.0 column is full height (top row has its block), the 0.0 column is empty at top
    assert rows[0][2] == "█" and rows[0][0] == " "
    # bottom row is filled for the non-zero columns
    assert rows[-1][1] != " " and rows[-1][2] != " "