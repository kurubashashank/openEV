from app.environment import WarehouseEnvironment
from app.graders import TaskGrader


def test_normalize_score_excludes_endpoints():
    assert 0.0 < WarehouseEnvironment.normalize_score(0.0) < 1.0
    assert 0.0 < WarehouseEnvironment.normalize_score(1.0) < 1.0


def test_grade_task_average_reward_is_strictly_in_range():
    result = TaskGrader.grade_task(
        task_id="easy",
        num_episodes=2,
        policy_fn=TaskGrader.forecast_policy,
        seed=42,
    )

    assert 0.0 < result["average_reward"] < 1.0
    assert all(0.0 < reward < 1.0 for reward in result["episode_rewards"])
