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
  --session <dir>     aggregate across all round_<NN>_<opp>/ subdirs of a multi-round play session: the
                      cumulative win-rate meter (every game of every round), the per-round summary, and the
                      latest round's live gameplay (the single-run view, reused). Supports --once / --save too.
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
from rich.markup import escape                      # noqa: E402
from rich.panel import Panel                       # noqa: E402
from rich.table import Table                       # noqa: E402
from rich.text import Text                         # noqa: E402

from sf2.eval import tui_model as T                # noqa: E402

REFRESH_HZ = 4
BAR_W = 26
TREND_W = 24
_DOT = "·"              # middot track for games not yet played

_VERDICT_STYLE = {"good": "bold green", "ok": "yellow", "bad": "bold red",
                  "not_scorable": "dim", None: "dim"}


def _conf_color(p: float) -> str:
    """System-1 confidence -> rich color: green high, yellow mid, red low."""
    if p >= 0.6:
        return "green"
    if p >= 0.35:
        return "yellow"
    return "red"


# --------------------------------------------------------------------------- panels

def _hp_line(name: str, hp: int, frac: float) -> Text:
    """Name + a shaded, health-colored bar + 'hp/176' label (label color tracks health)."""
    bar = T.shaded_bar(frac, BAR_W)                 # already colored by health level
    label_col = T._bar_color(max(0.0, min(1.0, frac)))
    return Text.from_markup(
        f"[bold]{escape(f'{name:<8}')}[/bold]{bar}  "
        f"[{label_col}]{hp:>4}/{T.HP_MAX}[/{label_col}]"
    )


def _trend_line(m: T.DashboardModel) -> Text:
    """One cell per GAME played, left->right in game order: green win / red loss / yellow tie.

    As wins accumulate the strip fills green toward the right. A trailing dim '·' track shows the
    games not yet played so the strip's length is stable across the run.
    """
    cell = {"W": "[green]█[/green]", "L": "[red]█[/red]", "T": "[yellow]█[/yellow]"}
    played = "".join(cell[v] for v in m.per_game_wl)
    remaining = max(0, m.games - len(m.per_game_wl))
    track = f"[dim]{_DOT * remaining}[/dim]" if remaining else ""
    return Text.from_markup(f"[bold]history [/bold]{played}{track}")


def _winrate_line(m: T.DashboardModel) -> Text:
    """Cumulative win-rate meter: a green shaded bar + 'win rate W/N PCT%'."""
    wins, played, pct = T.win_rate(m.per_game_wl)
    bar = T.shaded_bar((pct / 100.0) if played else 0.0, TREND_W, color="green")
    return Text.from_markup(f"[bold]winrate [/bold]{bar}  "
                            f"[green]win rate {wins}/{played} {pct}%[/green]")


def _left_panel(m: T.DashboardModel) -> Panel:
    title = Text.from_markup(
        f"[bold cyan]{escape(m.me)}[/bold cyan]  vs  [bold red]{escape(m.opp)}[/bold red]"
        f"    [white]game {m.cur_game + 1}/{m.games}  round {m.cur_round + 1}/{m.rounds}[/white]"
    )

    hp = Group(
        _hp_line(m.me, m.me_hp, m.me_hp_frac),
        _hp_line(m.opp, m.opp_hp, m.opp_hp_frac),
    )

    trend = Group(_trend_line(m), _winrate_line(m))

    pipe = Table.grid(padding=(0, 1))
    pipe.add_column(justify="right", style="dim", no_wrap=True)       # situation
    pipe.add_column(no_wrap=True)                                     # category
    pipe.add_column(no_wrap=True)                                     # action
    pipe.add_column(style="dim", no_wrap=True)                        # pressed
    pipe.add_row(Text.from_markup("[dim]SITUATION ->[/dim]"),
                 Text.from_markup("[cyan]CATEGORY ->[/cyan]"),
                 Text.from_markup("[white]ACTION ->[/white]"),
                 Text.from_markup("[dim]PRESSED[/dim]"))
    if not m.decisions:
        pipe.add_row("(waiting for first decision...)", "", "", "")
    for d in m.decisions:
        fb = " [bold red]fb![/bold red]" if d.fireball else ""
        sit = Text.from_markup(f"{escape(d.rng)}/{escape(d.opp_doing)} dx={d.dx:+d}{fb}")
        ccol = _conf_color(d.cat_prob)
        mcol = _conf_color(d.move_prob)
        cat = Text.from_markup(f"[cyan]{escape(d.category)}[/cyan] [{ccol}]{d.cat_prob:.0%}[/{ccol}]")
        mark = "ok" if d.follows_rule else "x"
        rcol = "green" if d.follows_rule else "red"
        act = Text.from_markup(
            f"[bold white]{escape(d.action)}[/bold white] [{mcol}]{d.move_prob:.0%}[/{mcol}]"
            f" [{rcol}][{escape(d.rule)}:{mark}][/{rcol}]"
        )
        pipe.add_row(sit, cat, act, Text(d.pressed, style="dim"))

    s = m.stats
    fr_col = "green" if s.follows_rule_pct >= 80 else "yellow"
    stats = Text.from_markup(
        f"this round  hits [bold]{s.hits}[/bold]  dealt [green]{s.dealt}[/green]  "
        f"taken [red]{s.taken}[/red]  ([dim]{s.decisions} decisions[/dim])\n"
        f"follows-rule so far  [{fr_col}]{s.follows_rule_pct:.1f}%[/{fr_col}]"
    )

    body = Group(title, Text(""), hp, Text(""), trend, Text(""), pipe, Text(""), stats)
    return Panel(body, title="LIVE GAMEPLAY", border_style="cyan")


def _memory_panel(m: T.DashboardModel) -> Panel:
    t = Text()
    t.append(f"{len(m.in_play)} rule(s) in play\n\n", style="bold green")
    if not m.in_play:
        t.append("(none)", style="dim")
    for i, rule in enumerate(m.in_play, 1):
        t.append(f"{i}. ", style="dim")
        t.append(f"{rule}\n", style="bright_cyan")      # short-memory rules in a readable accent
    return Panel(t, title="SHORT MEMORY (in-play rules)", border_style="green")


def _qwen_panel(m: T.DashboardModel) -> Panel:
    rows = Table.grid(padding=(0, 1))
    rows.add_column(no_wrap=True)
    if m.qwen_thinking:
        rows.add_row(Text("...thinking... (reflecting before the next game)",
                          style="bold yellow blink"))
        rows.add_row(Text(""))
    recent = m.qwen[-8:]
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


def _trend_panel(m: T.DashboardModel) -> Panel:
    """A bar graph that CLIMBS as the loop learns: one column per game, bar height = cumulative win-rate,
    plus a per-game hp-margin sparkline underneath. (The headline 'does it go up?' view.)"""
    wl = m.per_game_wl
    series = T.cum_winrate_series(wl)
    body = Text()
    if series:
        for row in T.vbars(series, height=5, lo=0.0, hi=1.0):
            body.append(row + "\n", style="green")
        body.append("0%" + " " * max(0, len(series) - 6) + "100%->\n", style="dim")
    else:
        body.append("(no games yet)\n\n\n\n\n", style="dim")
    wins, played, pct = T.win_rate(wl)
    body.append(f"win-rate {wins}/{played} = {pct}%\n", style="bold green")
    margins = T.margin_series(m.results)
    if margins:
        body.append("margin  ", style="dim")
        body.append(T.spark(margins), style="cyan")
        body.append(f"  last {margins[-1]:+d}", style="dim")
    return Panel(body, title="TREND (win-rate climbing)", border_style="green")


_PLAY_NODES = ("VISION", "TEXT", "CATEGORY", "MOVE", "CONTROL")
_LEARN_NODES = ("laya_text", "QWEN", "PLAYBOOK", "MEMORY")


def _flow_line(label: str, nodes, pulse: Optional[int], color: str) -> Text:
    """One path as nodes joined by arrows; the pulsing node is reverse-bright, its trail colored, the
    rest dim. pulse=None renders the whole path idle (dim). A small ◉ marks where the event is firing."""
    t = Text()
    t.append(f"{label:<10}", style="bold")
    for i, n in enumerate(nodes):
        if i:
            t.append(" → ", style="dim")
        if pulse is None:
            t.append(n, style="dim")
        elif i == pulse:
            t.append(f"◉{n}", style=f"bold reverse {color}")
        elif i == pulse - 1:
            t.append(n, style=color)
        else:
            t.append(n, style="dim")
    return t


def _pipeline_panel(thinking: bool, frame: int) -> Panel:
    """The cross-system data flow with a firing pulse. While playing, the pulse travels the PLAY path
    (vision -> text -> category -> move/short-memory -> control); while Qwen reflects between games, it
    travels the LEARN path (laya_text -> qwen -> playbook -> short-memory). The idle path is dim."""
    play_pulse = None if thinking else frame % len(_PLAY_NODES)
    learn_pulse = (frame % len(_LEARN_NODES)) if thinking else None
    body = Group(
        _flow_line("PLAY", _PLAY_NODES, play_pulse, "cyan"),
        Text(""),
        _flow_line("LEARN", _LEARN_NODES, learn_pulse, "magenta"),
        Text.from_markup("[dim]MEMORY feeds back into MOVE — the short memory the player reads.[/dim]"),
    )
    status = "reflecting (LEARN firing)" if thinking else "playing (PLAY firing)"
    return Panel(body, title=f"DATA FLOW — {status}", border_style="yellow")


def render(m: T.DashboardModel, frame: int = 0, with_footer: bool = True) -> Layout:
    """Single-run view. with_footer adds the TREND + DATA FLOW row; the session view sets it False and
    supplies its own career-wide trend and flow instead (so they are not duplicated)."""
    top = Layout()
    # wider right column, and SHORT MEMORY is the dominant panel there so more rules show at once
    top.split_row(Layout(name="left", ratio=1), Layout(name="right", ratio=1))
    top["left"].update(_left_panel(m))
    top["right"].split_column(
        Layout(_memory_panel(m), name="mem", ratio=3),
        Layout(_qwen_panel(m), name="qwen", ratio=2),
    )
    if not with_footer:
        return top
    layout = Layout()
    layout.split_column(Layout(top, name="top", ratio=5), Layout(name="bottom", ratio=1))
    layout["bottom"].split_row(
        Layout(_trend_panel(m), name="trend", ratio=2),
        Layout(_pipeline_panel(m.qwen_thinking, frame), name="flow", ratio=3),
    )
    return layout


# --------------------------------------------------------------------------- session (multi-round) view

def _session_trend_line(sm: T.SessionModel) -> Text:
    """The 'watch it grow' history strip: one cell per GAME across the WHOLE session, left->right in order
    (green win / red loss / yellow tie)."""
    cell = {"W": "[green]█[/green]", "L": "[red]█[/red]", "T": "[yellow]█[/yellow]"}
    played = "".join(cell[v] for v in sm.per_game_wl)
    return Text.from_markup(f"[bold]history [/bold]{played or '[dim](no games yet)[/dim]'}")


def _session_winrate_line(sm: T.SessionModel) -> Text:
    """Cumulative win-rate meter across every game of every round so far."""
    bar = T.shaded_bar((sm.cum_pct / 100.0) if sm.cum_played else 0.0, TREND_W, color="green")
    return Text.from_markup(f"[bold]winrate [/bold]{bar}  "
                            f"[green]win rate {sm.cum_wins}/{sm.cum_played} {sm.cum_pct}%[/green]")


def _session_panel(sm: T.SessionModel) -> Panel:
    live = Text.from_markup(
        f"[bold]SESSION[/bold]  live: [bold cyan]round {sm.active_num} vs "
        f"{escape(sm.active_opp)}[/bold cyan]  "
        f"[white]round {len(sm.rounds)}/{sm.rounds_planned or len(sm.rounds)}[/white]"
        + (f"   [dim]plan: {escape(' -> '.join(sm.opp_order))}[/dim]" if sm.opp_order else "")
    )
    rounds = Table.grid(padding=(0, 1))
    rounds.add_column(no_wrap=True)
    if not sm.rounds:
        rounds.add_row(Text("(no rounds yet)", style="dim"))
    for rsum in sm.rounds:
        marker = " [cyan](live)[/cyan]" if rsum.num == sm.active_num else ""
        rounds.add_row(Text.from_markup(
            f"round [bold]{rsum.num:>2}[/bold] vs [red]{escape(rsum.opp)}[/red]  "
            f"[green]{rsum.wins}[/green]-[red]{rsum.losses}[/red]"
            + (f"-[yellow]{rsum.ties}T[/yellow]" if rsum.ties else "")
            + f"   [dim]cum {rsum.cum_wins}/{rsum.cum_played} {rsum.cum_pct}%[/dim]{marker}"
        ))
    body = Group(live, Text(""), _session_trend_line(sm), _session_winrate_line(sm), Text(""), rounds)
    return Panel(body, title="SESSION (all rounds)", border_style="cyan")


def _career_trend(sm: T.SessionModel) -> Panel:
    """Career-wide trend: cumulative win-rate across EVERY game of every opponent so far, as a climbing
    bar chart (one column per game) + the running percent. The 'is it getting better over the ladder?' view."""
    series = T.cum_winrate_series(sm.per_game_wl)
    body = Text()
    if series:
        for row in T.vbars(series, height=5, lo=0.0, hi=1.0):
            body.append(row + "\n", style="green")
    else:
        body.append("(no games yet)\n\n\n\n\n", style="dim")
    body.append(f"career win-rate {sm.cum_wins}/{sm.cum_played} = {sm.cum_pct}%", style="bold green")
    return Panel(body, title="CAREER TREND (win-rate over the ladder)", border_style="green")


def render_session(sm: T.SessionModel, frame: int = 0) -> Layout:
    """Session view: the cumulative summary + career trend on top, the live round's single-run view in the
    middle, and the cross-system DATA FLOW pulse at the bottom."""
    layout = Layout()
    layout.split_column(
        Layout(name="head", ratio=2),
        Layout(name="active", ratio=3),
        Layout(name="flow", ratio=1),
    )
    layout["head"].split_row(Layout(_session_panel(sm), ratio=3), Layout(_career_trend(sm), ratio=2))
    thinking = bool(sm.active and sm.active.qwen_thinking)
    if sm.active is not None:
        layout["active"].update(render(sm.active, frame, with_footer=False))
    else:
        layout["active"].update(Panel(Text("(no active round yet)", style="dim"),
                                      title="LIVE GAMEPLAY", border_style="cyan"))
    layout["flow"].update(_pipeline_panel(thinking, frame))
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


def _launch_loop(spec: str, blank: bool = False):
    """Shell out to play_loop_screen.py and return (popen, run_dir_guess). READ-ONLY afterwards.
    With ``blank``, start from an EMPTY short memory (an empty --carry file) so the Coach builds the
    playbook from scratch -- the purest self-learning demo. Needs the Qwen server (SF2_QWEN_URL)."""
    parts = spec.split()
    if len(parts) != 4:
        sys.exit('--run expects "<me> <opp> <games> <rounds>"')
    me, opp, games, rounds = parts
    name = f"mon_{me}_{opp}_{int(time.time())}"
    run_dir = os.path.join("rollouts", "loop_screen", name)
    cmd = [sys.executable, os.path.join("scripts", "play_loop_screen.py"),
           "--me", me, "--opp", opp, "--games", games, "--rounds", rounds,
           "--name", name,
           "--cat-advisor", "runs/text_laya/cat_v3",
           "--move-advisor", "runs/text_laya/move_v2",
           "--no-score"]
    if blank:
        from sf2.system1.advice import char_menu_moves
        from sf2.system2.rule_entry import default_kit
        os.makedirs(os.path.join(REPO, run_dir), exist_ok=True)
        carry = os.path.join(run_dir, "blank_start.json")
        with open(os.path.join(REPO, carry), "w") as f:
            import json as _json
            _json.dump(default_kit(char_menu_moves(me)), f)   # blank = 1 attack foothold; block stays the default
        cmd += ["--carry", carry]
    env = dict(os.environ)  # passes SF2_QWEN_URL / SF2_ROM through unchanged
    proc = subprocess.Popen(cmd, cwd=REPO, env=env)
    return proc, run_dir


def _launch_career(me: str):
    """Launch the continuous blank-start career (scripts/play_career.py) for ME and return (proc,
    session_dir) so the TUI session view can tail it live. READ-ONLY afterwards. Needs SF2_QWEN_URL."""
    name = f"career_{me}_{int(time.time())}"
    session_dir = os.path.join("rollouts", "career", name)
    cmd = [sys.executable, os.path.join("scripts", "play_career.py"), "--me", me, "--name", name]
    env = dict(os.environ)
    proc = subprocess.Popen(cmd, cwd=REPO, env=env)
    return proc, session_dir


def _renderable(run_dir: str, grade: bool, session: bool, frame: int = 0):
    """Build the frame for ``run_dir``: the session view (aggregated across round dirs) when ``session``,
    else the single-run view. Both reuse the same single-run rendering for the live gameplay. ``frame``
    advances the data-flow pulse animation."""
    if session:
        return render_session(T.build_session_model(run_dir, grade_qwen=grade), frame)
    return render(T.build_model(run_dir, grade_qwen=grade), frame)


def _save_frame(run_dir: str, path: str, grade: bool, session: bool = False) -> int:
    """Render ONE frame to an image file, fully headless (no live terminal).

    A recording Console at a fixed 120-col width with a dark background captures the frame, then
    writes .svg (console.save_svg) or .html (console.save_html), chosen by the path extension.
    Returns 0 on success; a clear message + non-zero on an unsupported extension or a write error.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".svg", ".html", ".htm"):
        print(f"--save: unsupported extension '{ext}' (use .svg or .html)", file=sys.stderr)
        return 2
    renderable = _renderable(run_dir, grade, session, frame=2)  # mid-pulse so the snapshot looks alive
    with open(os.devnull, "w") as sink:
        console = Console(record=True, width=120, file=sink)   # fixed width, headless
        console.print(renderable)
        try:
            if ext == ".svg":
                console.save_svg(path, title="SF2 loop monitor")
            else:
                console.save_html(path)
        except OSError as exc:
            print(f"--save: could not write {path}: {exc}", file=sys.stderr)
            return 1
    return 0


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only live monitor for the self-learning loop.")
    ap.add_argument("--watch", metavar="RUN_DIR", help="attach to this run dir")
    ap.add_argument("--session", metavar="SESSION_DIR",
                    help="aggregate across all round_<NN>_<opp>/ subdirs of a multi-round play session dir "
                         "(cumulative win rate + per-round summary + the live round's gameplay)")
    ap.add_argument("--latest", action="store_true", help="watch the newest rollouts/loop_screen/*")
    ap.add_argument("--run", metavar="SPEC", help='launch play_loop_screen.py "<me> <opp> <games> <rounds>" and watch')
    ap.add_argument("--blank", action="store_true", help="with --run: start from a BLANK playbook (empty short memory; the Coach builds it)")
    ap.add_argument("--career", metavar="ME", nargs="?", const="chunli",
                    help="launch the CONTINUOUS blank-start career (scripts/play_career.py) for ME (default chunli) "
                         "and watch it: play forever, replay on loss, beat an opponent and move on")
    ap.add_argument("--once", action="store_true", help="render a single frame and exit")
    ap.add_argument("--save", metavar="PATH", help="render ONE frame headless to PATH (.svg or "
                    ".html) and exit; no live terminal needed")
    ap.add_argument("--no-grade", action="store_true", help="skip the G5 rule grading")
    args = ap.parse_args(argv)

    proc = None
    session = bool(args.session) or bool(args.career)
    if args.career:
        proc, run_dir = _launch_career(args.career)
        for _ in range(80):                       # wait for the first round dir to appear
            if os.path.isdir(run_dir) and any(n.startswith("round_") for n in os.listdir(run_dir)):
                break
            time.sleep(0.25)
    elif args.run:
        proc, run_dir = _launch_loop(args.run, blank=args.blank)
        # wait briefly for the run dir to appear
        for _ in range(40):
            if os.path.isdir(run_dir):
                break
            time.sleep(0.25)
    elif args.session:
        run_dir = args.session
    else:
        run_dir = _resolve_run_dir(args)

    grade = not args.no_grade

    if args.save:
        rc = _save_frame(run_dir, args.save, grade, session)
        if proc is not None:
            proc.terminate()
        return rc

    if args.once:
        Console().print(_renderable(run_dir, grade, session, frame=2))
        if proc is not None:
            proc.terminate()
        return 0

    console = Console()
    frame = 0
    try:
        with Live(_renderable(run_dir, grade, session, frame),
                  console=console, refresh_per_second=REFRESH_HZ, screen=True) as live:
            while True:
                status = (T.build_session_model(run_dir, grade_qwen=grade).status if session
                          else T.build_model(run_dir, grade_qwen=grade).status)
                frame += 1                                   # advance the data-flow pulse each tick
                live.update(_renderable(run_dir, grade, session, frame))
                if proc is not None and proc.poll() is not None and status in ("done", "empty"):
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
