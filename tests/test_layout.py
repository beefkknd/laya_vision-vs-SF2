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
