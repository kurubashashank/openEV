from pathlib import Path

import yaml


def test_openenv_task_difficulties_use_standard_labels():
    spec = yaml.safe_load(Path("openenv.yaml").read_text())

    difficulties = [task["difficulty"] for task in spec["tasks"]]
    assert difficulties == ["easy", "medium", "hard"]


def test_openenv_grading_score_range_is_strict():
    spec = yaml.safe_load(Path("openenv.yaml").read_text())

    minimum, maximum = spec["grading"]["score_range"]
    assert 0.0 < minimum < 1.0
    assert 0.0 < maximum < 1.0


def test_openenv_declares_grade_endpoint():
    spec = yaml.safe_load(Path("openenv.yaml").read_text())

    grade_endpoint = spec["api"]["endpoints"]["grade"]
    assert grade_endpoint["method"] == "POST"
    assert grade_endpoint["path"] == "/grade"
