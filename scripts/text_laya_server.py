"""Text laya as a helper process for System 1 (which runs in the torch venv). Runs in the laya-mlx venv; started by
sf2.system1.advisor.Advisor.

Per run (the default): one JSON request per stdin line, one JSON reply per stdout line.
    request  {"text": <situation + advice>, "question": <sf2.advice.question>}
    reply    {"probabilities": {move: p}}    or    {"error": "..."}
The first line it writes is {"ready": <checkpoint>} once the model is loaded.

Shared (--shared SOCKET): the same requests and replies over a unix socket, for every run using this checkpoint
(sf2.system1.shared_laya); it reserves --budget-gb in the jobs ledger before loading and exits after --idle seconds
with no client.
"""
import argparse
import json
import sys

import _path  # noqa: E402,F401
from sf2.config import JOBS_WAIT_S, TEXT_LAYA_IDLE_S  # noqa: E402
from sf2.eval.budget import Budget  # noqa: E402
from sf2.system1 import shared_laya, text_laya  # noqa: E402


def per_run(agent, label: str) -> None:
    out = sys.stdout
    sys.stdout = sys.stderr                     # nothing else may write to the reply channel
    out.write(json.dumps({"ready": label}) + "\n")
    out.flush()
    predict = _predict(agent)
    for line in sys.stdin:
        out.write(shared_laya.answer(predict, line))
        out.flush()


def _predict(agent):
    return lambda text, q: agent.predict(text, {"q": q})["answers"]["q"]["probabilities"]


def main(argv=None, load=text_laya.load, budget=Budget) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", help="a text laya checkpoint folder, or none for the base model")
    ap.add_argument("--shared", metavar="SOCKET", help="serve every client on this unix socket")
    ap.add_argument("--idle", type=float, default=TEXT_LAYA_IDLE_S, help="shared: exit after this long with no client")
    ap.add_argument("--budget-gb", type=float, default=0.0, help="shared: reserve this much in the jobs ledger")
    args = ap.parse_args(argv)
    adapter = args.checkpoint if args.checkpoint != "none" else None
    label = adapter or text_laya.BASE
    if not args.shared:
        per_run(load(adapter), label)
        return
    ledger = budget() if args.budget_gb > 0 else None
    rid = ledger.reserve(args.budget_gb, timeout=JOBS_WAIT_S) if ledger else None
    try:
        agent = load(adapter)
        shared_laya.serve(_predict(agent), args.shared, label, idle_s=args.idle)
    finally:
        if rid:
            ledger.release(rid)


if __name__ == "__main__":
    main()
