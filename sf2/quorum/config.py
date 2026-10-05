"""QuorumConfig: the one switch (``mode``) and every number the evolution loop may tune.

Modes (docs/plan_bee_quorum.md, rollout phases 0-2; Qwen is phase 3 and off unless ``qwen`` is set):
  shadow      text-laya plays exactly as today; the swarm votes beside it and is only LOGGED
  candidates  the move must come from the swarm's candidates; the table picks among them where it is confident,
              otherwise text-laya's pick stands (the hybrid, restricted to the candidates)
  vote        the weighted quorum decides: share >= theta acts; a split falls back to text-laya (or Qwen)
"""
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List

MODES = ("shadow", "candidates", "vote")
# flavour -> the round-1 categories that flavour may answer from (action_menu.CATEGORY_ORDER names).
# RETIRED 2026-10-05 (owner): the Option-2 category-forcing bees (defend/punish/combo). A (run A_20261005_084309)
# showed the force-combo bee HURT -- the quorum played combos ~6x/round and won 65% vs the ~71% baseline, because
# forcing a category the thick trunk already owned is blunt. The quorum is now organised to FILL the table's gaps
# (sf2/quorum/frontier.py): the 'frontier' bee explores the least-sampled followable move wherever the table is thin
# or blind, and the 'fireball' bee pushes the fb=1 slice A found most under-explored. laya (base_proposal) is the
# generalist base; the table voter exploits the thick trunk. The flavour machinery (voters.py) stays available for a
# config that sets its own ``flavors``, but the default is none.
FLAVORS: Dict[str, List[str]] = {}
VOTERS = ("laya", "table", "frontier", "fireball", "pressure")


@dataclass
class QuorumConfig:
    mode: str = "shadow"
    theta: float = 0.5           # quorum: act when the top move's share of the positive score is at least this
    epsilon: float = 0.05        # explore: act on the second candidate this often (vote / candidates modes)
    beta: float = 0.5            # recruitment strength (the table's good moves get a boost)
    gamma: float = 0.5           # cross-inhibition strength (the table's bad moves get suppressed)
    k: float = 8.0               # shrinkage: a cell mean counts n / (n + k) of its full weight
    eta: float = 0.1             # reliability learning rate (multiplicative weights)
    net_scale: float = 10.0      # squash a mean net hp m into (-1, 1) as m / (|m| + net_scale)
    table_min_n: int = 3         # samples before the table voter proposes a move in a cell
    w_min: float = 0.05          # reliability weight clamp
    w_max: float = 20.0
    qwen: bool = False           # escalate split votes to Qwen (needs an escalation hook in the driver)
    frontier: bool = True        # gap-filling bee: vote the least-sampled followable move where the table is thin
    fireball: bool = True        # gap-filling bee gated to fb=1 (the slice A found most under-explored)
    pressure: bool = True        # gap-filling bee gated to opp-attacking (the 'being-pressured' blind slice)
    priors: Dict[str, float] = field(default_factory=lambda: {v: 1.0 for v in VOTERS})
    flavors: Dict[str, List[str]] = field(default_factory=lambda: {k: list(v) for k, v in FLAVORS.items()})

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError("quorum mode %r is not one of %s" % (self.mode, ", ".join(MODES)))
        for name in ("theta", "epsilon"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError("quorum %s=%r must be in [0, 1]" % (name, v))
        for name in ("beta", "gamma", "eta"):
            if getattr(self, name) < 0:
                raise ValueError("quorum %s must be >= 0" % name)
        if self.k <= 0 or self.net_scale <= 0 or not 0 < self.w_min <= self.w_max:
            raise ValueError("quorum k, net_scale must be > 0 and 0 < w_min <= w_max")
        missing = [v for v in VOTERS if v not in self.priors]
        if missing:
            raise ValueError("quorum priors missing %s" % ", ".join(missing))

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "QuorumConfig":
        known = cls.__dataclass_fields__
        unknown = sorted(set(d) - set(known))
        if unknown:
            raise ValueError("unknown quorum config keys: %s" % ", ".join(unknown))
        base = cls().to_dict()
        if "priors" in d:                                  # a partial priors map keeps the other voters' defaults
            base["priors"].update(d["priors"])
        base.update({k: v for k, v in d.items() if k != "priors"})
        return cls(**base)

    @classmethod
    def load(cls, path: str) -> "QuorumConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
