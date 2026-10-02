"""The screen reader. Docstrings may mention value_oracle and sf2_bridge.lua: documentation is not use."""


def read(frame):
    facts = {"my_bar": "full"}
    return facts["my_bar"]
