#!/usr/bin/env python3
"""LIVE TEXT monitor for the self-learning loop -- a READ-ONLY tail of the files the loop flushes.

This is PURELY A MONITOR. It never touches the play path, the runner, or the hard gate: it imports
only ``sf2.eval.tui_model`` (the pure parser) and ``rich`` for rendering, and it only READS the run
dir. The optional ``--run`` mode shells out to ``scripts/play_loop_screen.py`` as a subprocess and
then watches its output dir -- it still only reads files; it does not drive the loop.

Layout (owner's design):
  LEFT  = live gameplay: title (ME vs OPP, game/round) + two HP bars + the decision pipeline
          (situation -> category pick -> action pick -> pressed, last ~6) + round/session STATS.
  RIGHT TOP   = SHORT MEMORY: the current in-play rules.
  RIGHT LOWER = QWEN: its per-game churn (added/removed), G5-graded when the scorer is importable,
          and a "...thinking..." banner inferred from file state between games.

Modes:
  --watch <run_dir>   attach to an existing/active run dir and render live.
  --latest            watch the newest rollouts/loop_screen/* dir.
  --run "<me> <opp> <games> <rounds>"
                      launch play_loop_screen.py (with the text-laya advisors + --no-score, passing
                      SF2_QWEN_URL / SF2_ROM from env) AND watch it.
  --once              render a single frame and exit (testing / no live terminal).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rich.console import Console, Group            # noqa: E402
from rich.layout import Layout                     # noqa: E402
from rich.live import Live                         # noqa: E402
from rich.panel import Panel                       # noqa: E402
from rich.table import Table                       # noqa: E402
from rich.text import Text                         # noqa: E402

from sf2.eval import tui_model as T                # noqa: E402

REFRESH_HZ = 4
BAR_W = 28

_VERDICT_STYLE = {"good": "bold green", "ok": "yellow", "bad": "bold red",
                  "not_scorable": "dim", None: "dim"}


# --------------------------------------------------------------------------- panels

def _hp_line(name: str, hp: int, frac: float, color: str) -> Text:
    bar = T.hp_bar(frac, BAR_W)
    t = Text()
    t.append(f"{name:<8}", style="bold")
    t.append(bar, style=color)
    t.append(f"  {hp:>3}/{T.HP_MAX}", style=color)
    return t


def _left_panel(m: T.DashboardModel) -> Panel:
    title = Text()
    title.append(f"{m.me}", style="bold cyan")
    title.append("  vs  ")
    title.append(f"{m.opp}", style="bold red")
    title.append(f"    game {m.cur_game + 1}/{m.games}  round {m.cur_round + 1}/{m.rounds}",
                 style="white")

    hp = Group(
        _hp_line(m.me, m.me_hp, m.me_hp_frac, "green"),
        _hp_line(m.opp, m.opp_hp, m.opp_hp_frac, "magenta"),
    )

    pipe = Table.grid(padding=(0, 1))
    pipe.add_column(justify="right", style="dim", no_wrap=True)       # situation
    pipe.add_column(no_wrap=True)                                     # category
    pipe.add_column(no_wrap=True)                                     # action
    pipe.add_column(style="dim", no_wrap=True)                        # pressed
    pipe.add_row("SITUATION ->", "CATEGORY ->", "ACTION ->", "PRESSED")
    if not m.decisions:
        pipe.add_row("(waiting for first decision...)", "", "", "")
    for d in m.decisions:
        fb = " fb!" if d.fireball else ""
        sit = f"{d.rng}/{d.opp_doing} dx={d.dx:+d}{fb}"
        cat = Text(f"{d.category} {d.cat_prob:.0%}")
        act = Text(f"{d.action} {d.move_prob:.0%}")
        mark = "ok" if d.follows_rule else "x"
        act.append(f" [{d.rule}:{mark}]",
                   style="green" if d.follows_rule else "red")
        pipe.add_row(sit, cat, act, d.pressed)

    s = m.stats
    stats = Text()
    stats.append(f"this round  hits {s.hits}  dealt {s.dealt}  taken {s.taken}  "
                 f"({s.decisions} decisions)\n")
    stats.append(f"follows-rule so far  {s.follows_rule_pct:.1f}%", style="cyan")

    body = Group(title, Text(""), hp, Text(""), pipe, Text(""), stats)
    return Panel(body, title="LIVE GAMEPLAY", border_style="cyan")


def _memory_panel(m: T.DashboardModel) -> Panel:
    t = Text()
    t.append(f"{len(m.in_play)} rule(s) in play\n\n", style="bold")
    if not m.in_play:
        t.append("(none)", style="dim")
    for i, rule in enumerate(m.in_play, 1):
        t.append(f"{i}. ", style="dim")
        t.append(f"{rule}\n")
    return Panel(t, title="SHORT MEMORY (in-play rules)", border_style="green")


def _qwen_panel(m: T.DashboardModel) -> Panel:
    rows = Table.grid(padding=(0, 1))
    rows.add_column(no_wrap=True)
    if m.qwen_thinking:
        rows.add_row(Text("...thinking... (reflecting before the next game)",
                          style="bold yellow blink"))
        rows.add_row(Text(""))
    recent = m.qwen[-4:]
    if not recent:
        rows.add_row(Text("(no qwen activity yet)", style="dim"))
    for qv in recent:
        head = Text(f"game {qv.game}: ", style="bold")
        if not qv.added and not qv.removed:
            head.append("no change", style="dim")
        rows.add_row(head)
        for gr in qv.added:
            line = Text("  + ", style="green")
            line.append(gr.line)
            if gr.verdict is not None:
                v = gr.verdict
                extra = ""
                if gr.value is not None and gr.forward is not None:
                    extra = f"  (val {gr.value:+.2f} vs fwd {gr.forward:+.2f})"
                line.append(f"  [{v}]{extra}", style=_VERDICT_STYLE.get(v, "dim"))
            rows.add_row(line)
        for rm in qv.removed:
            line = Text("  - ", style="red")
            line.append(rm, style="strike")
            rows.add_row(line)
    footer = Text(f"\nstatus: {m.status}", style="dim")
    return Panel(Group(rows, footer), title="QWEN (short-memory churn)", border_style="magenta")


def render(m: T.DashboardModel) -> Layout:
    layout = Layout()
    layout.split_row(Layout(name="left", ratio=3), Layout(name="right", ratio=2))
    layout["left"].update(_left_panel(m))
    layout["right"].split_column(
        Layout(_memory_panel(m), name="mem", ratio=1),
        Layout(_qwen_panel(m), name="qwen", ratio=2),
    )
    return layout


# --------------------------------------------------------------------------- run modes

def _resolve_run_dir(args) -> str:
    if args.watch:
        return args.watch
    if args.latest:
        d = T.latest_run_dir()
        if not d:
            sys.exit("no run dirs under rollouts/loop_screen/")
        return d
    sys.exit("need --watch <run_dir>, --latest, or --run \"<me> <opp> <games> <rounds>\"")


def _launch_loop(spec: str):
    """Shell out to play_loop_screen.py and return (popen, run_dir_guess). READ-ONLY afterwards."""
    parts = spec.split()
    if len(parts) != 4:
        sys.exit('--run expects "<me> <opp> <games> <rounds>"')
    me, opp, games, rounds = parts
    name = f"mon_{me}_{opp}_{int(time.time())}"
    run_dir = os.path.join("rollouts", "loop_screen", name)
    cmd = [sys.executable, os.path.join("scripts", "play_loop_screen.py"),
           "--me", me, "--opp", opp, "--games", games, "--rounds", rounds,
           "--name", name,
           "--cat-advisor", "runs/text_laya/cat_v1",
           "--move-advisor", "runs/text_laya/move_v1",
           "--no-score"]
    env = dict(os.environ)  # passes SF2_QWEN_URL / SF2_ROM through unchanged
    proc = subprocess.Popen(cmd, cwd=REPO, env=env)
    return proc, run_dir


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only live monitor for the self-learning loop.")
    ap.add_argument("--watch", metavar="RUN_DIR", help="attach to this run dir")
    ap.add_argument("--latest", action="store_true", help="watch the newest rollouts/loop_screen/*")
    ap.add_argument("--run", metavar="SPEC", help='launch play_loop_screen.py "<me> <opp> <games> <rounds>" and watch')
    ap.add_argument("--once", action="store_true", help="render a single frame and exit")
    ap.add_argument("--no-grade", action="store_true", help="skip the G5 rule grading")
    args = ap.parse_args(argv)

    proc = None
    if args.run:
        proc, run_dir = _launch_loop(args.run)
        # wait briefly for the run dir to appear
        for _ in range(40):
            if os.path.isdir(run_dir):
                break
            time.sleep(0.25)
    else:
        run_dir = _resolve_run_dir(args)

    grade = not args.no_grade

    if args.once:
        m = T.build_model(run_dir, grade_qwen=grade)
        Console().print(render(m))
        if proc is not None:
            proc.terminate()
        return 0

    console = Console()
    try:
        with Live(render(T.build_model(run_dir, grade_qwen=grade)),
                  console=console, refresh_per_second=REFRESH_HZ, screen=True) as live:
            while True:
                m = T.build_model(run_dir, grade_qwen=grade)
                live.update(render(m))
                if proc is not None and proc.poll() is not None and m.status in ("done", "empty"):
                    live.update(render(m))
                    break
                time.sleep(1.0 / REFRESH_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
