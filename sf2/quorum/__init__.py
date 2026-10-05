"""Bee-quorum System 1 (docs/plan_bee_quorum.md, docs/quorum_integration.md).

Several cheap voters propose a move -- the existing two-stage text-laya pick (the generalist BASE bee), the same
checkpoints asked from two Option-2 "flavours" (defend = block/move anti-pressure; punish = punch/special/throw/combo
openings -- aimed at the regions the 71% table OVERLOOKED), and the value table -- and a weighted tally decides. Each vote is weighted by
the voter's track record in this 'when' (multiplicative weights), plus recruitment (the table's good moves) and
cross-inhibition (its bad ones). A share >= theta acts (System 1); below it there is no quorum and the decision
falls back to text-laya, or to Qwen when an escalation hook is given (System 2).

Modules (all pure except the injected advisors / Qwen):
  config      QuorumConfig: the mode switch (shadow | candidates | vote) and every tunable number (the genome)
  tally       Proposal, the table voter, scoring, vote share
  voters      the three flavour voters over the two text-laya checkpoints
  reliability per-(when, voter) weights: update from outcomes, merge across parallel workers
  decider     quorum_decider(...) -> decide(moment), the drop-in for loop_runner.play_round(decide=...)
  escalate    make_qwen_pick(ask_qwen): the optional System-2 hook for split votes
  genome      bounds / sampling / mutation / fitness for the evolution loop (scripts/quorum_evolve.py)
Play-path safe like value_table: stdlib + sf2.system1.{advice, action_menu, screen_words, value_table} + sf2.vocab.
"""
