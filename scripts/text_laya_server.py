"""Text laya as a helper process for System 1 (which runs in the torch venv): one JSON request per stdin line,
one JSON reply per stdout line. Runs in the laya-mlx venv; started by sf2.advisor.Advisor.

    request  {"text": <situation + advice>, "question": <sf2.advice.question>}
    reply    {"probabilities": {move: p}}    or    {"error": "..."}
The first line it writes is {"ready": <checkpoint>} once the model is loaded.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sf2 import text_laya  # noqa: E402


def main() -> None:
    adapter = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "none" else None
    agent = text_laya.load(adapter)
    out = sys.stdout
    sys.stdout = sys.stderr                     # nothing else may write to the reply channel
    out.write(json.dumps({"ready": adapter or text_laya.BASE}) + "\n")
    out.flush()
    for line in sys.stdin:
        try:
            req = json.loads(line)
            ans = agent.predict(req["text"], {"q": req["question"]})["answers"]["q"]
            reply = {"probabilities": ans["probabilities"]}
        except Exception as e:                  # a bad request must not kill the game: report it
            reply = {"error": "%s: %s" % (type(e).__name__, e)}
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
