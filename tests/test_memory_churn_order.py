"""memory_churn.diff lists added / dropped lessons in the memory's own order, the same on every run (it used set
order, so the churn report reshuffled between runs: found 2026-09-29 comparing report.py with memory_churn.py)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = """
from sf2.system2.memory_churn import diff
old = {"lessons": [{"text": t, "kind": "avoid"} for t in ("avoid lp up close", "avoid sweep far away", "avoid throw up close")]}
new = {"lessons": [{"text": t, "kind": "use_more"} for t in ("use more c.mk at mid range", "use more hk far away", "use more mp up close")]}
c = diff(old, new, ["lp", "sweep", "throw", "c.mk", "hk", "mp"])
print(c.added, c.dropped)
"""


def run(seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    return subprocess.run([sys.executable, "-c", CODE], cwd=HERE, env=env, capture_output=True, text=True,
                          check=True).stdout


def test_same_order_whatever_the_hash_seed():
    outs = {run(s) for s in range(6)}
    assert outs == {"['use more c.mk at mid range', 'use more hk far away', 'use more mp up close'] "
                    "['avoid lp up close', 'avoid sweep far away', 'avoid throw up close']\n"}
