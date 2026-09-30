"""A stand-in for text laya in the shared-server tests: deterministic probabilities from the text and the options,
plus trigger words that make it die (DIE) or never answer (HANG). FAKE_LAYA_DELAY slows every answer (so concurrent
requests overlap); FAKE_LAYA_STARTS (a file) gets one line per model load, so a test can count servers."""
import hashlib
import os
import time


def probabilities(text, question):
    opts = list(question["criteria"])
    w = {o: int(hashlib.sha1((text + "|" + o).encode()).hexdigest()[:8], 16) + 1 for o in opts}
    total = sum(w.values())
    return {o: w[o] / total for o in opts}


class FakeAgent:
    def predict(self, text, questions):
        if "DIE" in text:
            os._exit(3)
        if "HANG" in text:
            time.sleep(3600)
        time.sleep(float(os.environ.get("FAKE_LAYA_DELAY", "0")))
        return {"answers": {"q": {"probabilities": probabilities(text, questions["q"])}}}


def load(adapter_dir=None, dtype="float32"):
    starts = os.environ.get("FAKE_LAYA_STARTS")
    if starts:
        with open(starts, "a") as f:
            f.write("%d\n" % os.getpid())
    time.sleep(float(os.environ.get("FAKE_LAYA_LOAD_S", "0")))
    return FakeAgent()
