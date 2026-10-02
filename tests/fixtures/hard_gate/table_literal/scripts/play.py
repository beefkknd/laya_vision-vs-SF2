import json
import os

import _path  # noqa: F401

TABLE = os.path.join("lessons", "value_oracle_v1.json")


def main():
    with open(TABLE) as f:
        return json.load(f)
