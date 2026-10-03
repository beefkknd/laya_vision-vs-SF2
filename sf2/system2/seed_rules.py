"""Seed the lesson loop from the owner's web/AB research (lessons/book.json) - the loop's STARTING playbook for Qwen.

The research is already done (scripts/book.py build, per-opponent verified single-line fixed-advice arms that helped vs
no advice). This module only LOADS it: for one opponent it takes the VERIFIED lines (``lines``; ``not_verified`` is left
out), validates each against text laya's play-time parser so a line the grammar can no longer read is dropped - never
crashes the loop - and returns the survivors as registry entries (sf2.system2.lessons.from_book) tagged provenance
"web" so they rotate in the registry like any other rule. It does not run the loop; ``dry_run`` / ``python -m
sf2.system2.seed_rules`` prints, per opponent, the seed lessons it would admit.

Validation (per line, via sf2.system1.advice.read, the same parser System 1 uses at play time):
  1. the stored claim renders to the line (the grammar text laya reads - what scripts/book.py also checks at build);
  2. advice.read re-derives that same claim from the line, to a move System 1 can actually follow.
A line that fails either is dropped and the reason logged (WARNING); a loader that silently kept an unreadable line
would feed text laya advice it cannot act on (docs/component_boundaries.md).

Provenance: the registry evidence already carries a ``source`` field; from_book sets it to "players' tip". Per the
owner's framing (web/AB research) these seed entries set ``source="web"`` - a new value in that existing field - and
add ``book_sha256`` (which bytes they came from). The claim keeps ``view="book"`` (the registry-admitted view the loop
and lesson_prompt rely on) and the state stays "verified" (from_book's contract: in play first, its A/B evidence is the
proof the per-situation history cannot reproduce).
"""
import argparse
import hashlib
import json
import logging
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from ..system1 import advice as A
from . import lessons as L

log = logging.getLogger(__name__)

SOURCE = "web"                 # provenance tag written into evidence["source"] (see module docstring)
DEFAULT_ME = "chunli"
# text laya's polarity -> the registry's claim kind (the inverse of scripts/book.py's KIND).
POLARITY_KIND = {"soft": "use_more", "hard": "always", "neg": "avoid"}


def _claim_from_line(line: str, moves: Sequence[str]) -> Optional[Dict]:
    """Re-derive the claim from the line with text laya's play-time parser (never raises). None when it names no move
    System 1 can follow."""
    les = A.read(line, moves)
    if les.move is None or les.polarity not in POLARITY_KIND:
        return None
    return {"kind": POLARITY_KIND[les.polarity], "move": les.move, "range": les.where, "when": les.when}


def _reads_as(got: Dict, stored: Dict, moves: Sequence[str]) -> bool:
    """Does the play-parser's reading ``got`` agree with the book's stored claim? Exact on every field, with one
    accepted reconciliation: the book may store the generic throw word ("throw"/"grab") where the play parser
    canonicalizes it to the menu's concrete throw move (advice.throw_move) - the throw alias (advice.THROW_WORD, owner
    2026-10-02). That one substitution names the SAME move System 1 follows, so it is not a drift; any other
    disagreement (a different move, kind, range or condition) is."""
    if L.key(got) == L.key(stored):
        return True
    sm = stored.get("move")
    return bool(isinstance(sm, str) and A.THROW_WORD.fullmatch(sm) and got["move"] == A.throw_move(moves)
                and got["kind"] == stored["kind"]
                and got.get("range") == stored.get("range") and got.get("when") == stored.get("when"))


def validate_tip(tip: Dict, moves: Sequence[str]) -> Tuple[bool, str]:
    """Is a book tip admissible? (ok, reason-if-dropped). It must render from its stored claim AND parse back - via
    advice.read against the SAME two-stage move menu text laya reads at play time (``moves`` = char_menu_moves(me)) -
    to the move System 1 would actually follow, so text laya reads it as System 2 wrote it. The book's generic throw
    word reconciles to the menu's concrete throw move (see ``_reads_as``)."""
    line, stored = tip.get("line"), tip.get("claim")
    if not isinstance(line, str) or not isinstance(stored, dict):
        return False, "tip has no line/claim"
    try:
        rendered = L.render(stored)
    except (KeyError, TypeError) as e:
        return False, "stored claim does not render (%s)" % e
    if rendered != line:
        return False, "stored claim renders to %r, not the line %r" % (rendered, line)
    got = _claim_from_line(line, moves)
    if got is None:
        return False, "advice.read parses no followable move in %r" % line
    if not _reads_as(got, stored, moves):
        return False, "advice.read reads %r as %r, not the stored claim %r" % (line, got, stored)
    if A.stance_unreliable(got["move"]):
        return False, ("stance-unreliable move %r (crouch c.* / jump j.|jf.*) voids to block when keyed "
                       "on his state - dropped from the seed" % got["move"])
    return True, ""


def load_book(path: str) -> Tuple[Dict, str]:
    """The whole book document and the sha256 of its exact bytes."""
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _retag(entry: Dict, book: str, sha: str) -> Dict:
    """A new entry whose evidence is tagged provenance ``SOURCE`` and carries the book's bytes (immutable: copies)."""
    ev = dict(entry["evidence"], source=SOURCE, book=book, book_sha256=sha)
    return dict(entry, evidence=ev)


def seed_lessons(path: str, opp: str, me: Optional[str] = None, game: int = -1) -> List[Dict]:
    """One opponent's verified book lines as starting registry entries (tagged "web"); [] when the opponent is not in
    the book. Each line is validated; a line that no longer parses is dropped with a logged reason."""
    doc, sha = load_book(path)
    # Validate against the character's full two-stage MOVE MENU - the exact names text laya chooses from at play time
    # (char_menu_moves(me), incl. hadoken_hp / shoryuken_hp / throw_F+hp) - NOT the Stage-1 action vocabulary
    # (system1.choices), which lacks the two-stage names and carries a bare "hp" that misreads "throw_F+hp" as "hp".
    moves = A.char_menu_moves(me or doc.get("me", DEFAULT_ME))
    tips = doc.get("opponents", {}).get(opp, {}).get("lines", [])     # verified only; not_verified never read
    good = []
    for t in tips:
        ok, why = validate_tip(t, moves)
        if not ok:
            log.warning("seed_rules: dropping %s book line %r: %s", opp, t.get("line"), why)
            continue
        good.append(t)
    reg = L.from_book(good, path, game)         # verified entries, deduped and ordered as the loop admits them
    return [_retag(r, path, sha) for r in reg]


def dry_run(path: str, opponents: Optional[Sequence[str]] = None) -> int:
    """Print, per opponent, the seed lessons the loader would admit."""
    doc, sha = load_book(path)
    opps = list(opponents) if opponents else sorted(doc.get("opponents", {}))
    print("%s (sha256 %s), provenance %r" % (path, sha[:12], SOURCE))
    for opp in opps:
        reg = seed_lessons(path, opp)
        print("%-8s %d seed lesson(s): %s" % (opp, len(reg), "; ".join(r["line"] for r in reg) or "(none)"))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Dry-run the web/AB seed playbook (sf2.system2.seed_rules).")
    ap.add_argument("--book", default="lessons/book.json")
    ap.add_argument("--opp", action="append", help="limit to this opponent (repeatable); default: all in the book")
    args = ap.parse_args(argv)
    return dry_run(args.book, args.opp)


if __name__ == "__main__":
    sys.exit(main())
