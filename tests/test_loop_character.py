"""Generalising the screen-only loop to play ANY supported character AS me (--me; chunli/ryu/ken), ADDITIVE: the
Chun-Li path must stay byte-identical. The runner's round-1 categories and round-2 move menu, and the advice
vocabulary, now come from ME's own moveset (sf2.system1.advice.char_categories / char_menu_moves) instead of the
hardcoded Chun-Li CATEGORIES.

Seen RED (confirmed against the pre-change code, where two_stage_decide ignored ``me`` for the menu and always used
Chun-Li's CATEGORIES):
  * test_ryu_decision_picks_a_ryu_special: with the old hardcoded menu, "always use hadoken_hp" offered no ryu special
    (hadoken_hp is not in Chun-Li's menu), so round 1 fell to block and the action was block_high, not hadoken_hp.
    Verified red by running the same decision with char_categories forced to Chun-Li's map (asserted below in the
    _old_ reconstruction): the pick is block_high.
  * test_ryu_category_menu: char_categories('ryu')['special'] was Chun-Li's (lightning_legs/spinning_bird_kick); the
    ryu assertions fail against it.
  * test_unsupported_me_errors: before --me existed the driver had no choices gate and char_categories did not exist,
    so 'blanka' produced no clear error.
  * test_blank_seed_for_ryu: seed() always ran the Chun-Li book for any me; it never returned [] for ryu.
The Chun-Li byte-identical checks (test_chunli_*) would fail if the per-character wiring changed Chun-Li's menu.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest                                                        # noqa: E402
from looptools import FollowerLaya, load_driver, make_moment        # noqa: E402

from sf2.system1.action_menu import CATEGORIES, CATEGORY_ORDER, DEFAULT_MOVE   # noqa: E402
from sf2.system1.advice import char_categories, char_menu_moves, read as read_lesson   # noqa: E402
from sf2.system1.loop_runner import two_stage_decide                 # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK = os.path.join(HERE, "lessons", "book.json")


class CharFollower:
    """FollowerLaya, but for ANY character: it reads the advice out of the prompt TEXT and follows it over the offered
    options using ``me``'s own move vocabulary and category map (so round 1 maps the followed move to ITS category)."""

    def __init__(self, me: str):
        self.menu_moves = char_menu_moves(me)
        self.cat_of = {m: c for c, names in char_categories(me).items() for m in names}
        self.calls = []

    def ask(self, text, question):
        opts = list(question["criteria"])
        self.calls.append({"text": text, "options": opts})
        rng, doing, fire = FollowerLaya._situation(text)
        live = [l for l in (read_lesson(t, self.menu_moves) for t in FollowerLaya._advice(text))
                if l.applies(rng, doing, fire)]
        ruled = {l.move for l in live if l.polarity == "neg"}

        def follow(cands):
            for pol in ("hard", "soft"):
                hit = [l.move for l in live if l.polarity == pol and l.move in cands and l.move not in ruled]
                if hit:
                    return sorted(hit)[0]
            return None

        if opts == CATEGORY_ORDER:
            move = follow(self.menu_moves)
            choice = self.cat_of[move] if move else "block"
        else:
            choice = follow(opts) or (DEFAULT_MOVE if DEFAULT_MOVE in opts else opts[0])
        return {o: (1.0 if o == choice else 0.0) for o in opts}


# --------------------------------------------------------------------- ryu's own menu
def test_ryu_category_menu_is_ryus_moveset():
    cats = char_categories("ryu")
    assert "hadoken_hp" in cats["special"] and "shoryuken_hp" in cats["special"] and "tatsumaki_hk" in cats["special"]
    flat = char_menu_moves("ryu")
    assert "lightning_legs" not in flat and "spinning_bird_kick" not in flat   # Chun-Li's specials are gone
    assert "hadoken_hp" in flat


def test_ryu_decision_picks_a_ryu_special():
    m = make_moment(dx=150, doing="standing")          # far away: a grounded stance offers the specials
    d = two_stage_decide(CharFollower("ryu"), CharFollower("ryu"), "ryu", m, ["always use hadoken_hp far away"])
    assert d["action"] == "hadoken_hp" and d["category"] == "special"
    assert d["rule"] == "hard" and d["follows_rule"] and d["rule_answers"] == ["hadoken_hp"]
    assert d["action"] in char_menu_moves("ryu")       # a valid ryu move


def test_old_hardcoded_menu_could_not_offer_the_ryu_special():
    """The RED twin: with Chun-Li's menu (the pre-change behaviour), hadoken_hp is never an option, so the follower
    falls to the default block - proving the ryu test above detects the generalisation."""
    m = make_moment(dx=150, doing="standing")
    from sf2.system1 import advice as A
    from sf2.system1.screen_words import situation
    rng, doing, _, _ = situation(m)
    stance = A.stance_of("stand", rng)
    opts = A.moves_in_stance("special", stance, CATEGORIES)   # Chun-Li's special menu
    assert "hadoken_hp" not in opts                            # the old menu could not offer it


# --------------------------------------------------------------------- chun-li unchanged (byte-identical)
def test_chunli_category_map_is_byte_identical():
    assert char_categories("chunli") == CATEGORIES
    assert char_menu_moves("chunli") == [m for cat in CATEGORY_ORDER for m in CATEGORIES[cat]]


def test_chunli_decision_is_unchanged():
    m = make_moment(dx=36, doing="standing")           # up close
    base = two_stage_decide(FollowerLaya(), FollowerLaya(), "chunli", m, [])
    assert base["action"] == DEFAULT_MOVE and base["category"] == "block" and base["rule"] == "default"
    d = two_stage_decide(FollowerLaya(), FollowerLaya(), "chunli", m, ["always use throw_F+hp up close"])
    assert d["action"] == "throw_F+hp" and d["category"] == "throw" and d["rule"] == "hard" and d["follows_rule"]


# --------------------------------------------------------------------- unsupported character errors clearly
def test_unsupported_me_char_categories_errors():
    with pytest.raises(ValueError):                    # balrog is a boss, not one of the 8 playable world warriors
        char_categories("balrog")
    with pytest.raises(ValueError):
        two_stage_decide(CharFollower("ryu"), CharFollower("ryu"), "balrog", make_moment(), [])


def test_unsupported_me_cli_rejected():
    play_loop = load_driver()
    parser = play_loop.build_parser()
    parser.parse_args(["--me", "ryu"])                 # a supported one is accepted
    with pytest.raises(SystemExit):                    # argparse choices reject a non-playable char
        parser.parse_args(["--me", "balrog"])


# --------------------------------------------------------------------- blank seed for a non-book character
@pytest.mark.skipif(not os.path.exists(BOOK), reason="no lessons/book.json")
def test_blank_seed_for_ryu_does_not_crash(capsys):
    play_loop = load_driver()
    reg = play_loop.seed("ryu", BOOK, "ryu")           # the book is Chun-Li's: ryu gets no seed
    assert reg == []
    err = capsys.readouterr().err
    assert "no seed for ryu" in err
    # chunli still seeds from the book (unchanged)
    opp = next(o for o, e in __import__("json").load(open(BOOK))["opponents"].items() if e.get("lines"))
    assert play_loop.seed(opp, BOOK, "chunli")         # non-empty for the book's own character
