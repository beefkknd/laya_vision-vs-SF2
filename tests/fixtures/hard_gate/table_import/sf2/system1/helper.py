from ..data.value_oracle import rank


def decide(table, text):
    return rank(table, text, [])
