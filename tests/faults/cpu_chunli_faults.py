"""Seeded faults for the CPU Chun-Li games (docs/prereg_movement_data.md, "Addition: CPU Chun-Li games"): provenance,
the P2-Chun-Li character check, sampling and the stop rule on Chun-Li only, her act set's builder, and the gate's
labels / provenance / coverage / clock. Each mutates one line, runs the tests, and restores the file. Every fault must
turn the tests red; exit 1 if one stays green.

    python tests/faults/cpu_chunli_faults.py
"""
import os
import signal
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TIMEOUT = 180
TESTS = ["tests/test_cpu_chunli.py", "tests/test_action_collect.py", "tests/test_action_data.py"]
K, COL, IO, DC, GATE, MG = ("sf2/data/cpu_chunli.py", "sf2/data/action_collect.py", "sf2/data/action_collect_io.py",
                            "sf2/data/action_data_cpu.py", "sf2/data/action_gate.py", "sf2/data/movement_gate.py")
FAULTS = [
    # 1 provenance
    ("prov: Chun-Li accepted as player 1", K, "if p1_char not in FIGHTERS or p1_char == CPU:",
     "if p1_char not in FIGHTERS:"),
    ("prov: a wrong value not reported", K, "        elif rec[k] != want[k]:", "        elif False:"),
    ("prov: a missing field not reported", K, 'out.append("provenance %s missing" % k)', "continue"),
    ("prov: expected controller from the constant, not the run", K, '"controller": run["controller"],',
     '"controller": CONTROLLER,'),
    ("prov: player 1 not checked against the run", K, '    if p1_char not in run.get("p1_chars", []):',
     "    if False:"),
    ("prov: run settings name another slot", K, "collection=COLLECTION, chunli_slot=SLOT,",
     'collection=COLLECTION, chunli_slot="p1",'),
    ("prov: compare lists old codes as new", K, '"new_vs_mv3": [c for c in with_rows if c not in old]',
     '"new_vs_mv3": with_rows'),
    # 2 the P2-Chun-Li check
    ("chars: player 2 never checked", IO, "for p, want in sorted(ids.items()):",
     "for p, want in sorted(ids.items())[:1]:"),
    ("chars: wrong characters committed", IO, "        if wrong:\n", "        if False:\n"),
    ("chars: empty game passes", IO, '        return ["no rows"]', "        return []"),
    # 3 sampling and the stop rule on Chun-Li only
    ("sample: every actor sampled", COL, "if self.sample_actors is not None and actor not in self.sample_actors:",
     "if False:"),
    ("sample: unknown sample actor accepted", COL,
     "if sample_actors is not None and not set(sample_actors) <= set(actors.values()):", "if False:"),
    ("sample: the loop does not pass the actors to the sampler", IO,
     'ImageSaver(os.path.join(base, "images"), game), ring, keep)', 'ImageSaver(os.path.join(base, "images"), game), ring)'),
    ("stop: novelty counts every actor", IO, "and (actors is None or p[\"actor\"] in actors)}",
     "}"),
    ("stop: advance ignores the actors", IO, "new = novel(pairs, prog.seen, actors)", "new = novel(pairs, prog.seen)"),
    ("stop: resume ignores the actors", IO, 'by_game.get(g["game"], []), actors)', 'by_game.get(g["game"], []))'),
    ("stop: provenance for another player 1 accepted", IO,
     'if provenance is not None and provenance.get("p1_char", actors[1]) != actors[1]:', "if False:"),
    ("io: provenance not on the pairs", IO, "pairs = [dict(p, **prov) for p in sampler.finish()]",
     "pairs = sampler.finish()"),
    ("io: provenance not on games.jsonl", IO, "summary or {}, **prov, game=game,", "summary or {}, game=game,"),
    # 4 the builder
    ("build: a pair of player 1 accepted", DC, "if (p.get(\"actor\"), p.get(\"player\")) != (K.CPU, PLAYER):",
     "if False:"),
    ("build: pair provenance not checked", DC, 'bad = ["%s: %s" % (where, x) for x in K.provenance_problems(p, want)]',
     "bad = []"),
    ("build: collector label not checked", DC,
     'if lab is None or (lab[0].code, lab[1]) != (p["code"], p["stg"]):', "if lab is None:"),
    ("build: asked about her as me", DC, '("act", act_question(K.CPU, PLAYER, codes), A.code_name(code))',
     '("act", act_question(K.CPU, 1, codes), A.code_name(code))'),
    ("build: the note is Chun-Li's", DC, '"state_text": note,', '"state_text": U.eye_note(K.CPU),'),
    ("build: rows without provenance", DC, "**{k: p[k] for k in KEEP}, **K.provenance(p1))",
     "**{k: p[k] for k in KEEP})"),
    ("build: the stage row dropped", DC, '                           ("stage", stage_question(K.CPU, PLAYER), "stg%d" % stg)):',
     "                           ):"),
    # 5 the gate
    ("gate: player-1 rows in the CPU set pass", GATE, '                        if r.get("player") not in players:',
     "                        if False:"),
    ("gate: CPU set checks both players", GATE, "    players = (2,) if _is_cpu(meta) else (1, 2)",
     "    players = (1, 2)"),
    ("gate: CPU set's actor names swapped", GATE, 'chars = {1: o, 2: "chunli"} if _is_cpu(meta)',
     'chars = {1: "chunli", 2: o} if _is_cpu(meta)'),
    ("gate: no provenance gate", GATE, '        gates["provenance"] = provenance_check(meta, files)\n',
     "        pass\n"),
    ("gate: state_text not checked", GATE, '                if r.get("state_text") != want["note"]:',
     "                if False:"),
    ("gate: no run.json passes", GATE, 'return {"pass": False, "error": "no run.json in %s" % meta["root"]}',
     'return {"pass": True, "error": "no run.json in %s" % meta["root"]}'),
    ("gate: provenance mismatches not counted", GATE, "                if probs:\n                    bad += 1",
     "                if probs:\n                    bad += 0"),
    ("gate: coverage counts player 1's codes", GATE,
     '        obs = collections.Counter({k: v for k, v in obs.items() if k[0] == meta["cpu"]})', "        pass"),
    ("gate: CPU clock read by the blue mask", GATE, "MIN_AGREE_DISC, seed,\n                                             clock=\"digits\")",
     "MIN_AGREE_DISC, seed,\n                                             clock=\"blue\")"),
    ("clock: digit palette loses the orange strokes", MG,
     "HUD_DIGITS = np.array([(255, 156, 107), (247, 107, 66), (24, 66, 173)])",
     "HUD_DIGITS = np.array([(24, 66, 173), (0, 189, 189)])"),
    ("clock: digits read as any blue", MG, "return (im[..., None, :] == HUD_DIGITS).all(-1).any(-1)",
     "return (im[..., 2] > 150) & (im[..., 0] < 60)"),
]


def run(fault) -> bool:
    name, path, old, new = fault
    full = os.path.join(ROOT, path)
    with open(full) as f:
        src = f.read()
    if src.count(old) != 1:
        raise SystemExit("fault %r: the line to mutate is not found once in %s" % (name, path))
    try:
        with open(full, "w") as f:
            f.write(src.replace(old, new))
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"] + TESTS,
                               cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT)
            red = r.returncode != 0
        except subprocess.TimeoutExpired:
            red = True
    finally:
        with open(full, "w") as f:
            f.write(src)
    return red


def main() -> int:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))     # a kill still restores the mutated file
    only = sys.argv[1:]
    missed = 0
    faults = [f for f in FAULTS if not only or any(o in f[0] for o in only)]
    for fault in faults:
        red = run(fault)
        missed += not red
        print("%-5s %s" % ("RED" if red else "GREEN", fault[0]), flush=True)
    print("%d of %d faults caught" % (len(faults) - missed, len(faults)))
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
