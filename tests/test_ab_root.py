"""ab_memory's output folder (harness ledger #21, 2026-09-30): two batches launched in the same second wrote into one
rollouts/ab/<stamp> folder and interleaved their logs (529 broken lines, the Ken expert batch lost)."""
import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ab():
    sys.path.insert(0, os.path.join(HERE, "scripts"))
    spec = importlib.util.spec_from_file_location("ab_memory", os.path.join(HERE, "scripts", "ab_memory.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_two_batches_in_the_same_second_get_their_own_folders(tmp_path):
    m = ab()
    a = m.run_root(41001, base=str(tmp_path), stamp="20260930-053734")
    b = m.run_root(41002, base=str(tmp_path), stamp="20260930-053734")
    assert a != b and os.path.isdir(a) and os.path.isdir(b)


def test_an_existing_folder_is_never_shared(tmp_path):
    m = ab()
    m.run_root(41001, base=str(tmp_path), stamp="20260930-053734")
    with pytest.raises(SystemExit, match="exists"):
        m.run_root(41001, base=str(tmp_path), stamp="20260930-053734")
