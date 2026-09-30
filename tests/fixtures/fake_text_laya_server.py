"""scripts/text_laya_server.py's real command line with the fake text laya (tests/fixtures/fake_laya.py)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts")]

import fake_laya  # noqa: E402
import text_laya_server  # noqa: E402

if __name__ == "__main__":
    text_laya_server.main(sys.argv[1:], load=fake_laya.load)
