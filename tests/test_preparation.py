import json
from types import SimpleNamespace

import pytest

from joint_ttopt.preparation import collect_entry, declared_envelopes, entry_normalization
from test_interfaces import EnvironmentFixture


def test_collection_produces_native_labels_and_training_provenance(tmp_path):
    renderer = SimpleNamespace(encode=lambda *args: (1, 2), tokenizer=None)
    teacher = SimpleNamespace(select=lambda *args: 0, metadata={"label_source": "software_fixture"})
    specification = {"split": "entry_training", "task_id": "training_task",
                     "partner_schedule_id": "training_partners"}
    labels, episodes = collect_entry(EnvironmentFixture(), teacher, renderer, specification, tmp_path,
                                      [1, 2], fallback_spans={0: 1})
    assert len(labels) == len(episodes) == 2
    assert all(row["split"] == "entry_training" and row["action"] in row["legal_actions"] for row in labels)
    assert all(row["input_ids"] == [1, 2] for row in labels)
    assert json.loads((tmp_path / "metadata.json").read_text())["source_revision"] == "fixture-v1"
    normalization = json.loads((tmp_path / "normalization.json").read_text())
    assert normalization["agents"]["0"] == {"lower": 1.0, "upper": 1.0, "fallback_span": 1.0}


def test_collection_excludes_evaluation_split(tmp_path):
    with pytest.raises(ValueError, match="entry-training"):
        collect_entry(EnvironmentFixture(), None, None, {"split": "test"}, tmp_path, [0])
    with pytest.raises(ValueError, match="declared benchmark"):
        entry_normalization([{"complete": True, "rewards": {0: [1]}}])


def test_declared_envelope_is_computed_from_trajectory_bounds():
    result = declared_envelopes({"reference_scale": 1, "validity_record": "software_fixture", "agents": {
        "0": {"advantage_bound": 2, "score_bounds": [3], "state_bounds": [4],
              "span": 2, "regularization": 0.5, "horizon": 2, "discount": 1}}})
    assert result["sample_bounds"]["0"] == pytest.approx(8)
