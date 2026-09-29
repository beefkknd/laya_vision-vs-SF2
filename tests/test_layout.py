"""Package layout invariants: paths into the repo come from sf2.config.REPO, so moving a module never breaks them."""
import glob
import os

from sf2.config import REPO


def test_repo_is_the_checkout():
    assert os.path.exists(os.path.join(REPO, "pyproject.toml")) and os.path.isdir(os.path.join(REPO, "sf2"))


def test_only_config_finds_the_repo_from_its_own_file():
    users = [f for f in glob.glob("sf2/**/*.py", recursive=True)
             if f != os.path.join("sf2", "config.py") and "__file__" in open(f).read()]
    assert users == []


MACHINE = ("~/", "/Volumes/", "/Applications/", "/Users/", "http://127.0.0.1:8000")


def _code_strings(path):
    """String constants in code (docstrings left out: they may show example paths)."""
    import ast
    tree = ast.parse(open(path).read())
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]


def test_machine_paths_live_only_in_config():
    files = glob.glob("sf2/**/*.py", recursive=True) + glob.glob("scripts/*.py") + glob.glob("tools/*.py")
    found = ["%s: %r" % (f, s) for f in files if f != os.path.join("sf2", "config.py")
             for s in _code_strings(f) if s.startswith(MACHINE)]
    assert found == []


def test_env_overrides_reach_config():
    import subprocess
    import sys
    env = dict(os.environ, SF2_QWEN_URL="http://other:1/v1", SF2_MLX_PYTHON="/opt/mlx/bin/python",
               SF2_LAYA_VISION="runs/x/best")
    out = subprocess.run([sys.executable, "-c", "from sf2 import config as c; print(c.QWEN_URL, c.MLX_PYTHON, "
                          "c.LAYA_VISION)"], cwd=REPO, env=env, capture_output=True, text=True, check=True).stdout
    assert out.split() == ["http://other:1/v1", "/opt/mlx/bin/python", "runs/x/best"]


def test_port_ranges_do_not_overlap():                 # each runner uses base .. base + count - 1
    from sf2.config import PORTS
    spans = sorted((base, base + count) for base, count in PORTS.values())
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
