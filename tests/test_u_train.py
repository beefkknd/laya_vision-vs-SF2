"""scripts/train.py on the U data (docs/prereg_u_perception.md): the checkpoint records note v3 and "perception": true
(rows mixing notes refused), validation comes from the dirs' val.jsonl (whole held-out games, never a position split of
training games), and the stage-1 coverage checks (actions x ranges x postures) do not apply to perception dirs; the
share / sampling checks still do. No model is loaded here."""
import json
import os

import pytest

from sf2.data import train_data as TD
from sf2.data import u_data as U
from sf2.data.train_data import checkpoint_tags

V2 = "me=ryu dist=mid side=left dx=+65 my_bar=full opp_bar=full opp_airborne=0 opp_crouch=0 opp_attacking=1"


def ex(text, t="choice", dataset="chunli", i=0, img="frames/a_now.png"):
    return {"state": {"images": ["x", img], "context": text}, "q": {"t": t, "ins": "x", "crit": None},
            "dataset": dataset, "id": "r%d" % i}


def test_v3_rows_tag_perception():
    assert checkpoint_tags([ex("me=chunli"), ex("me=ryu")]) == {"note_version": 3, "value_questions": False,
                                                               "perception": True}


def test_v3_mixed_with_a_ram_note_is_refused():
    with pytest.raises(ValueError):
        checkpoint_tags([ex("me=chunli"), ex(V2)])


def test_v3_with_score_questions_is_refused():
    with pytest.raises(ValueError):
        checkpoint_tags([ex("me=chunli"), ex("me=chunli", "score")])


def test_old_tags_have_no_perception_key():
    assert "perception" not in checkpoint_tags([ex(V2)])


def u_dir(tmp_path, name, n_train=3, n_val=2):
    d = tmp_path / name
    os.makedirs(d)
    q = U.perception_question("range")

    def rec(i, split):
        return {"id": "%s-%s-%d" % (name, split, i), "images": ["frames/%s_%d_prev.png" % (split, i),
                                                                 "frames/%s_%d_now.png" % (split, i)],
                "state_text": U.eye_note(name), "question": q, "label": 1, "perception": True, "task": "perception"}
    for split, n in (("train", n_train), ("val", n_val)):
        with open(d / (split + ".jsonl"), "w") as f:
            f.write("".join(json.dumps(rec(i, split)) + "\n" for i in range(n)))
    return str(d)


def test_load_data_takes_validation_from_val_jsonl(tmp_path):
    dirs = [u_dir(tmp_path, "chunli"), u_dir(tmp_path, "ryu")]
    train, val = TD.load_data(dirs)
    assert len(train) == 6 and len(val) == 4
    assert {os.path.basename(e["state"]["images"][-1]).split("_")[0] for e in val} == {"val"}
    assert checkpoint_tags(train + val)["perception"] is True


def test_stage1_coverage_is_skipped_for_perception_dirs_but_shares_still_count(tmp_path):
    dirs = [u_dir(tmp_path, "chunli"), u_dir(tmp_path, "ryu")]
    train, val = TD.load_data(dirs)
    assert TD.coverage_problems(train, val, dirs) == []
    dirs2 = [u_dir(tmp_path / "b", "chunli", n_train=30), u_dir(tmp_path / "b", "ryu")]
    train, val = TD.load_data(dirs2)
    assert any("rows unequal" in p for p in TD.coverage_problems(train, val, dirs2))
