"""HARD GATE (owner rule 2026-10-02): "the 'table' and 'RAM' can only be used in testing, not in real code".

Pure and static: nothing here imports a repo module; every module is read as text and parsed with ``ast``.

REAL CODE = every repo module statically reachable by imports from ENTRY_POINTS (the play runners), plus every module
under EXTRA_ROOTS (sf2/screen/). An import anywhere in a module counts, including one inside a function: a lazy import
is still reachable. REAL CODE must not
  * import a FORBIDDEN_MODULES entry (TABLE, RAM, or TESTING-only code such as the replay scorer),
  * import a FORBIDDEN_SYMBOLS name from a mixed module (one that holds both RAM paths and RAM-free helpers),
  * contain a TABLE string (STRING_RULES; docstrings excluded: documentation is not use), an ``oracle=`` keyword,
  * touch a RAM attribute (RAM_ATTRS) or read a RAM row key (RAM_KEYS / RAM_KEY_PATTERNS) by subscript or ``.get``.
The RAM-free Lua bridge (SCREEN_LUA) must not read or write emulator memory (LUA_RULES).

Mixed modules are followed for their own imports (each edge is checked by the same rules) but their bodies are not
scanned: the RAM half lives there by design, and the gate is about what REAL CODE pulls from them.
ALLOW lists the reviewed exceptions, each with its reason. ``check`` returns the violations; empty = clean.
"""
import ast
import fnmatch
import os
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterator, List, Optional, Sequence, Set, Tuple

TABLE, RAM, TESTING, GATE = "TABLE", "RAM", "TESTING", "GATE"

# The play runners. Add the Qwen loop runner (README M1/G1) here the day it exists.
ENTRY_POINTS: Tuple[str, ...] = (
    "scripts/play_screen.py",   # screen-only play (S0): the current real runner
)
EXTRA_ROOTS: Tuple[str, ...] = (
    "sf2/screen",               # the screen reader: real code whether or not a runner imports it yet
)
SCREEN_LUA: Tuple[str, ...] = (
    "mesen/sf2_bridge_screen.lua",   # the RAM-free bridge screen-only play runs in Mesen
)

# (repo-relative path glob, kind, why). Importing any of these from REAL CODE is a violation.
FORBIDDEN_MODULES: Tuple[Tuple[str, str, str], ...] = (
    # TABLE
    ("sf2/data/value_oracle.py", TABLE, "the lookup table (lessons/value_oracle_v1.json loader / rank / cell)"),
    # RAM: the RAM map and RAM-driven emulator code
    ("sf2/emu/ram.py", RAM, "RAM map: Var / parse_map / load_map (ram_maps/sf2_snes.txt WRAM addresses)"),
    ("sf2/emu/boot.py", RAM, "arcade boot driven by RAM WATCH vars and SETUP RAM values"),
    # RAM: builders of notes / labels / outcomes from RAM rows
    ("sf2/data/vs_defense.py", RAM, "block probes and outcomes from RAM rows (p*_state, p*_y)"),
    ("sf2/data/vs_moves.py", RAM, "RAM p*_state codes and RAM-row predicates (seen, life_drops, connected, gap)"),
    ("sf2/data/pairs_labels.py", RAM, "movement / facing / air / distance labels from RAM rows"),
    ("sf2/data/perception.py", RAM, "perception labels from RAM rows (range_band, phase, air, projectile, can_act)"),
    ("sf2/system1/opp_moves.py", RAM, "opponent-move classifier over RAM rows (p2_state, shot2, p2_y)"),
    ("sf2/system1/game_log.py", RAM, "game log built from RAM rows (action_entry, game_entry, ram_entry)"),
    # TESTING-only: allowed to use table/RAM, never reachable from play
    ("tests/*", TESTING, "the test suite"),
    ("scripts/replay_score.py", TESTING, "offline replay scorer"),
    ("sf2/system1/screen_replay.py", TESTING, "offline replay scorer (uses the table)"),
    ("scripts/gate_*.py", TESTING, "gate script"),
    ("scripts/collect_*.py", TESTING, "collection script"),
    ("scripts/build_*.py", TESTING, "dataset build script"),
    ("scripts/probe_*.py", TESTING, "RAM probe script"),
    ("sf2/sprites/*", TESTING, "sprite collection from OAM/RAM"),
    ("sf2/data/build.py", TESTING, "data builder"),
    ("sf2/data/*collect*.py", TESTING, "data collection (RAM rows)"),
    ("sf2/data/*gate*.py", TESTING, "data gate"),
    ("sf2/data/*_data*.py", TESTING, "dataset builder (action/eye/movement/pairs/train/u/value data)"),
    ("sf2/data/*fill*.py", TESTING, "dataset fill"),
    ("sf2/data/*probe*.py", TESTING, "RAM probe"),
    ("sf2/data/pairs_*.py", TESTING, "pairs dataset pipeline"),
)

# Mixed modules: path -> {symbol: (kind, why)}. Other symbols from these modules are RAM-free and allowed.
FORBIDDEN_SYMBOLS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "sf2/emu/vs.py": {
        "VARS": (RAM, "WRAM addresses of both fighters, timer, result, projectiles"),
        "NAMES": (RAM, "RAM row key names"),
        "_FIELDS": (RAM, "per-fighter WRAM offsets"),
        "rows_of": (RAM, "Obs.rams -> RAM row dicts"),
        "view": (RAM, "RAM row p1_/p2_ -> a_/d_ keys"),
        "settled": (RAM, "neutral test on a RAM row"),
        "record": (RAM, "records an exchange as RAM rows"),
        "gap_state": (RAM, "walks to a gap measured in RAM x"),
        "boot_vs": (RAM, "VS boot checked against RAM (set_vars VARS)"),
        "Take": (RAM, "a recorded take: RAM rows"),
        "GROUND_Y": (RAM, "RAM y of the ground"),
        "START_X": (RAM, "RAM x of the round start"),
    },
    "sf2/data/vs_sweep.py": {
        "note": (RAM, "text note built from a RAM row"),
        "current_note": (RAM, "note rebuilt from a RAM record"),
        "outcome": (RAM, "action outcome from RAM rows"),
        "outcome_question": (RAM, "outcome question over RAM outcome classes"),
        "OUTCOMES": (RAM, "RAM outcome classes"),
        "OUTCOME_CRITERIA": (RAM, "RAM outcome criteria"),
        "mirror_record": (RAM, "mirrors a RAM record"),
        "split_of": (RAM, "train/test split of the RAM sweep (table cells)"),
        "TEST_INDEX": (TABLE, "held-out sweep index shared with the table"),
        "GAPS": (RAM, "setup gaps in RAM x"),
    },
    "sf2/emu/headless.py": {
        "BRIDGE": (RAM, "mesen/sf2_bridge.lua, the RAM bridge"),
        "bridge_for_port": (RAM, "writes a copy of the RAM bridge"),
    },
    "sf2/emu/mesen.py": {},     # transport for both bridges; its RAM methods are caught at the call site (RAM_ATTRS)
}

# Strings: (rule, kind, predicate on the literal, why).
STRING_RULES = (
    ("table-literal", TABLE, lambda s: "value_oracle" in s, "names the lookup table"),
    ("table-flag", TABLE, lambda s: "--oracle" in s, "the --oracle flag"),
    ("ram-bridge-literal", RAM, lambda s: os.path.basename(s) == "sf2_bridge.lua", "the RAM Lua bridge"),
    ("ram-lua-literal", RAM, lambda s: "snesWorkRam" in s, "Mesen work-RAM memory type"),
)
TABLE_KEYWORDS: FrozenSet[str] = frozenset({"oracle"})       # System1(..., oracle=...)

# Attribute names that are RAM access on a bridge / observation (sf2/emu/mesen.py MesenBridge, Obs).
RAM_ATTRS: Dict[str, str] = {
    "rams": "Obs.rams: RAM rows from the bridge",
    "set_vars": "MesenBridge.set_vars: choose RAM vars to stream",
    "dump_wram": "MesenBridge.dump_wram: whole work RAM",
    "poke": "MesenBridge.poke: write work RAM",
    "rows_of": "sf2.emu.vs.rows_of: RAM rows",
    "ram_entry": "sf2.system1.game_log.ram_entry: RAM rows in the log",
    "VARS": "sf2.emu.vs.VARS: WRAM addresses",
}

# RAM row keys, flagged when read by subscript or .get(...).
RAM_KEYS: FrozenSet[str] = frozenset({
    "timer", "result",                                  # sf2/emu/vs.py VARS globals
    "shot1", "shot1_x", "shot2", "shot2_x",             # sf2/emu/vs.py projectile slots
    # ram_maps/sf2_snes.txt names (sf2/emu/ram.py)
    "my_hp", "opp_hp", "my_life", "opp_life", "my_x", "opp_x", "my_y", "opp_y", "my_state", "opp_state",
    "my_react", "opp_react", "my_sub", "opp_sub", "my_dizzy", "opp_dizzy", "fireball", "fireball_x",
    "my_fireball", "my_fireball_x", "my_special", "my_facing",
    # RAM-derived game-log fields (sf2/system1/game_log.py action_entry, sf2/system1/opp_moves.py)
    "opp_air", "opp_reaction", "opp_attacked", "opp_blocked", "i_was_hit", "gap_after", "my_life_after",
    "opp_life_after", "opp_move", "opp_shot",
})
_FIGHTER_FIELDS = "hp|life|x|y|state|sub|react|dizzy|special|facing|char"
RAM_KEY_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^p[12]_(%s)$" % _FIGHTER_FIELDS),     # sf2/emu/vs.py VARS per fighter
    re.compile(r"^[ad]_(%s)$" % _FIGHTER_FIELDS),      # sf2/emu/vs.py view(): attacker / defender keys
)

# Lua in the RAM-free bridge: (rule, regex, why). Comments are stripped first.
LUA_RULES = (
    ("lua-work-ram", re.compile(r"\bsnesWorkRam\b"), "names Mesen work RAM"),
    ("lua-mem-read", re.compile(r"\bemu\.read\w*\s*\("), "emu.read*: memory read"),
    ("lua-mem-write", re.compile(r"\bemu\.write\w*\s*\("), "emu.write*: memory write"),
)

# Reviewed exceptions: (path, rule id) -> reason.
ALLOW: Dict[Tuple[str, str], str] = {
    ("sf2/system1/screen_emu.py", "ram-attr:rams"): "guard: raises RamForbidden if the bridge ever sends RAM rows",
}


@dataclass(frozen=True, order=True)
class Violation:
    path: str
    line: int
    kind: str
    rule: str
    detail: str

    def __str__(self) -> str:
        return "%s:%d: [%s] %s: %s" % (self.path, self.line, self.kind, self.rule, self.detail)


@dataclass(frozen=True)
class Policy:
    entry_points: Tuple[str, ...] = ENTRY_POINTS
    extra_roots: Tuple[str, ...] = EXTRA_ROOTS
    screen_lua: Tuple[str, ...] = SCREEN_LUA
    forbidden_modules: Tuple[Tuple[str, str, str], ...] = FORBIDDEN_MODULES
    forbidden_symbols: Dict[str, Dict[str, Tuple[str, str]]] = field(default_factory=lambda: FORBIDDEN_SYMBOLS)
    allow: Dict[Tuple[str, str], str] = field(default_factory=lambda: ALLOW)


@dataclass(frozen=True)
class Edge:
    line: int
    target: str          # repo-relative path (may not exist: a hypothetical module under a repo package)
    names: Tuple[str, ...]  # symbols imported from target ("from X import a, b"); () for a module import
    dotted: str


# ---------------------------------------------------------------- import resolution

def _rel(repo: str, path: str) -> str:
    return os.path.relpath(path, repo).replace(os.sep, "/")


def _module_file(base: str, parts: Sequence[str]) -> Optional[str]:
    """The file of module ``parts`` under ``base``: a/b.py or a/b/__init__.py when it exists; when it does not but its
    top-level package / module is in the repo, the hypothetical a/b.py (still checked against the forbidden lists);
    None for a module outside the repo (stdlib, third party)."""
    if not parts:
        return os.path.join(base, "__init__.py")
    stem = os.path.join(base, *parts)
    if os.path.isfile(stem + ".py"):
        return stem + ".py"
    if os.path.isfile(os.path.join(stem, "__init__.py")):
        return os.path.join(stem, "__init__.py")
    if os.path.isdir(os.path.join(base, parts[0])) or os.path.isfile(os.path.join(base, parts[0] + ".py")):
        return os.path.join(stem, "__init__.py") if os.path.isdir(stem) else stem + ".py"
    return None


def _up(d: str, n: int) -> str:
    for _ in range(n):
        d = os.path.dirname(d)
    return d


def _edges(repo: str, path: str, tree: ast.AST, search: Sequence[str], is_module) -> Iterator[Edge]:
    """Every import edge of a module. ``from X import a``: a is a submodule edge when X/a.py exists or is forbidden
    (``is_module(path)``), else a symbol of X."""
    here = os.path.dirname(os.path.join(repo, path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                f = next((f for f in (_module_file(b, a.name.split(".")) for b in search) if f), None)
                if f:
                    yield Edge(node.lineno, _rel(repo, f), (), a.name)
        elif isinstance(node, ast.ImportFrom):
            parts = node.module.split(".") if node.module else []
            bases = search if node.level == 0 else [_up(here, node.level - 1)]
            dotted = "." * node.level + (node.module or "")
            for base in bases:
                f = _module_file(base, parts)
                if not f:
                    continue
                stem = os.path.join(base, *parts)
                symbols = []
                for a in node.names:
                    sub = next((c for c in (stem + "/" + a.name + ".py", stem + "/" + a.name + "/__init__.py")
                                if os.path.isfile(c)), stem + "/" + a.name + ".py")
                    package = f.endswith("__init__.py") or not os.path.isfile(f)
                    if package and (os.path.isfile(sub) or is_module(_rel(repo, sub))):
                        yield Edge(node.lineno, _rel(repo, sub), (), "%s.%s" % (dotted, a.name))
                    else:
                        symbols.append(a.name)
                yield Edge(node.lineno, _rel(repo, f), tuple(symbols), dotted)
                break


# ---------------------------------------------------------------- content scan

def _docstrings(tree: ast.AST) -> Set[int]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def is_ram_key(s: str) -> bool:
    return s in RAM_KEYS or any(p.match(s) for p in RAM_KEY_PATTERNS)


def _key_of(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args \
            and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return node.args[0].value
    return None


def _mixed_aliases(tree: ast.AST, edges: List[Edge], mixed: Dict[str, Dict]) -> Dict[str, str]:
    """local name -> mixed module path, for ``import sf2.emu.vs as vs`` / ``from sf2.emu import vs``."""
    out = {}
    by_line = {(e.line, e.dotted.split(".")[-1]): e.target for e in edges if e.target in mixed and not e.names}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                t = by_line.get((node.lineno, a.name.split(".")[-1]))
                if t:
                    out[a.asname or a.name.split(".")[-1]] = t
    return out


def scan_source(path: str, tree: ast.AST, aliases: Dict[str, str],
                mixed: Dict[str, Dict[str, Tuple[str, str]]]) -> Iterator[Violation]:
    docs = _docstrings(tree)
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            for rule, kind, pred, why in STRING_RULES:
                if pred(node.value):
                    yield Violation(path, line, kind, rule, "%r (%s)" % (node.value[:60], why))
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in TABLE_KEYWORDS:
                    yield Violation(path, line, TABLE, "table-keyword", "%s=... passes the lookup table" % kw.arg)
        if isinstance(node, ast.Attribute):
            if node.attr in RAM_ATTRS:
                yield Violation(path, line, RAM, "ram-attr:%s" % node.attr, RAM_ATTRS[node.attr])
            if isinstance(node.value, ast.Name) and node.value.id in aliases:
                bad = mixed[aliases[node.value.id]].get(node.attr)
                if bad:
                    yield Violation(path, line, bad[0], "symbol:%s" % node.attr,
                                    "%s.%s (%s)" % (aliases[node.value.id], node.attr, bad[1]))
        key = _key_of(node)
        if key is not None and is_ram_key(key):
            yield Violation(path, line, RAM, "ram-key:%s" % key, "reads RAM row key %r" % key)


def scan_lua(path: str, text: str) -> Iterator[Violation]:
    text = re.sub(r"--\[(=*)\[.*?\]\1\]", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    for n, raw in enumerate(text.splitlines(), 1):
        code = raw.split("--", 1)[0]
        for rule, rx, why in LUA_RULES:
            if rx.search(code):
                yield Violation(path, n, RAM, rule, why)


# ---------------------------------------------------------------- the gate

def _forbidden(target: str, policy: Policy) -> Optional[Tuple[str, str]]:
    for pat, kind, why in policy.forbidden_modules:
        if fnmatch.fnmatchcase(target, pat):
            return kind, why
    return None


def roots(repo: str, policy: Policy) -> List[str]:
    out = list(policy.entry_points)
    for d in policy.extra_roots:
        full = os.path.join(repo, d)
        for dirpath, _, files in sorted(os.walk(full)):
            out += sorted(_rel(repo, os.path.join(dirpath, f)) for f in files if f.endswith(".py"))
    return out


def reach(repo: str, policy: Policy = Policy()) -> Tuple[Dict[str, Optional[str]], List[Violation]]:
    """Walk the import closure. Returns (reached module -> the module that first imported it, violations)."""
    search = [repo] + sorted({os.path.dirname(os.path.join(repo, e)) for e in policy.entry_points})
    mixed = policy.forbidden_symbols
    parent: Dict[str, Optional[str]] = {}
    found: List[Violation] = []
    queue = deque()
    for r in roots(repo, policy):
        if not os.path.isfile(os.path.join(repo, r)):
            found.append(Violation(r, 0, GATE, "missing-root", "entry point / root does not exist"))
        elif r not in parent:
            parent[r] = None
            queue.append(r)
    while queue:
        path = queue.popleft()
        try:
            with open(os.path.join(repo, path), encoding="utf-8") as f:
                tree = ast.parse(f.read(), path)
        except (OSError, SyntaxError, ValueError) as e:
            found.append(Violation(path, 0, GATE, "unreadable", "%s: %s" % (type(e).__name__, e)))
            continue
        edges = list(_edges(repo, path, tree, search, lambda t: _forbidden(t, policy) is not None))
        for e in edges:
            bad = _forbidden(e.target, policy)
            if bad:
                found.append(Violation(path, e.line, bad[0], "import:%s" % e.target,
                                       "imports %s (%s)%s" % (e.dotted, bad[1], _via(parent, path))))
                continue
            for s in e.names:
                sym = mixed.get(e.target, {}).get(s)
                if sym:
                    found.append(Violation(path, e.line, sym[0], "symbol:%s" % s,
                                           "from %s import %s (%s)%s" % (e.dotted, s, sym[1], _via(parent, path))))
            if os.path.isfile(os.path.join(repo, e.target)) and e.target not in parent:
                parent[e.target] = path
                queue.append(e.target)
        if path not in mixed:
            found.extend(scan_source(path, tree, _mixed_aliases(tree, edges, mixed), mixed))
    return parent, found


def _via(parent: Dict[str, Optional[str]], path: str) -> str:
    chain = []
    p = parent.get(path)
    while p is not None and len(chain) < 8:
        chain.append(p)
        p = parent.get(p)
    return " [via %s]" % " < ".join(chain) if chain else ""


def check(repo: str, policy: Policy = Policy()) -> List[Violation]:
    """Every violation of the hard gate in ``repo``, sorted; [] = clean."""
    _, found = reach(repo, policy)
    for lua in policy.screen_lua:
        full = os.path.join(repo, lua)
        if not os.path.isfile(full):
            found.append(Violation(lua, 0, GATE, "missing-root", "RAM-free Lua bridge does not exist"))
            continue
        with open(full, encoding="utf-8") as f:
            found.extend(scan_lua(lua, f.read()))
    kept = {v for v in found if (v.path, v.rule) not in policy.allow}
    return sorted(kept)
