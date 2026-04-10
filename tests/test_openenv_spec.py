from pathlib import Path

import yaml


def test_openenv_task_difficulties_are_strictly_in_range():
    spec = yaml.safe_load(Path("openenv.yaml").read_text())

    difficulties = [task["difficulty"] for task in spec["tasks"]]
    assert all(0.0 < difficulty < 1.0 for difficulty in difficulties)
