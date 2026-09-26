"""A stand-in for MesenBridge that replays a RAM trace: recorded from the real ROM, or written by hand.

A trace is a header plus one row per frame. Row 0 is the RAM right after the savestate loads; row i is the RAM
after frame i, with the buttons that were held on frame i. RAM is stored as raw bytes of a few WRAM windows, so
any RAM map whose addresses fall inside them can be decoded from the same trace. That is the point: a trace
recorded once from the ROM checks a new map, and the env's own inputs are checked against the recorded ones
(a facing bug fails as "frame 412: env pressed ['left'], trace recorded ['right']").

On disk (tests/fixtures/*.jsonl.gz): line 1 = header {"windows": [[start, length], ...], ...}, then one line per
row {"in": [buttons] or null, "mem": [hex bytes per window]}. ``"in": null`` means "not checked".
"""
import gzip
import json
from typing import Dict, List, Optional, Sequence

import numpy as np

from sf2.mesen import Obs


def load(path: str) -> Dict:
    with gzip.open(path, "rt") as f:
        header = json.loads(f.readline())
        return {"header": header, "rows": [json.loads(line) for line in f if line.strip()]}


def save(path: str, trace: Dict) -> None:
    with gzip.open(path, "wt") as f:
        f.write(json.dumps(trace["header"]) + "\n")
        for r in trace["rows"]:
            f.write(json.dumps(r) + "\n")


def make_trace(ram_map, values: Sequence[Dict[str, int]], inputs: Optional[Sequence] = None) -> Dict:
    """A hand-written trace: one window per map variable, ``values[i]`` = the variables after frame i."""
    windows = [[v.addr, v.size] for v in ram_map]
    rows = []
    for i, vals in enumerate(values):
        mem = [(vals[v.name] & ((1 << 8 * v.size) - 1)).to_bytes(v.size, "little").hex() for v in ram_map]
        rows.append({"in": sorted(inputs[i]) if inputs is not None else None, "mem": mem})
    return {"header": {"windows": windows, "synthetic": True}, "rows": rows}


class TraceMesen:
    def __init__(self, trace: Dict):
        self.windows = trace["header"]["windows"]
        self.rows = trace["rows"]
        self.vars = None
        self.t = 0

    def set_vars(self, specs):
        self.vars = list(specs)

    def _byte(self, row: Dict, addr: int) -> int:
        for (start, length), hexs in zip(self.windows, row["mem"]):
            if start <= addr < start + length:
                return int(hexs[2 * (addr - start):2 * (addr - start) + 2], 16)
        raise KeyError("address 0x%04X is outside the trace's windows" % addr)

    def _values(self, t: int) -> List[int]:
        row, out = self.rows[t], []
        for v in self.vars:
            n = int.from_bytes(bytes(self._byte(row, v.addr + k) for k in range(v.size)), "little")
            if v.signed and n >= 1 << (8 * v.size - 1):
                n -= 1 << (8 * v.size)
            out.append(n)
        return out

    @staticmethod
    def _img():
        return np.zeros((224, 256, 3), np.uint8)

    def load_state(self, data):
        self.t = 0
        return Obs([self._values(0)], [], {0: self._img()}, None)

    def run(self, frames, caps=()):
        rows, imgs = [self._values(self.t)], {i: self._img() for i in caps}
        for f in frames:
            self.t += 1
            assert self.t < len(self.rows), "trace ends at frame %d" % (len(self.rows) - 1)
            rec = self.rows[self.t]["in"]
            assert rec is None or sorted(f) == rec, \
                "frame %d: env pressed %s, trace recorded %s" % (self.t, sorted(f), rec)
            rows.append(self._values(self.t))
        return Obs(rows, [], imgs, None)

    def close(self):
        pass
