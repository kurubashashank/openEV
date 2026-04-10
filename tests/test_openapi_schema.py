from app.main import app


def test_openapi_declares_strict_bounds_for_all_score_fields():
    schema = app.openapi()["components"]["schemas"]

    step_reward = schema["StepInfo"]["properties"]["step_reward"]
    assert step_reward["exclusiveMinimum"] == 0.0
    assert step_reward["exclusiveMaximum"] == 1.0

    cumulative_reward = schema["StepResponse"]["properties"]["reward"]
    assert cumulative_reward["exclusiveMinimum"] == 0.0
    assert cumulative_reward["exclusiveMaximum"] == 1.0

    average_reward = schema["TaskGradeResponse"]["properties"]["average_reward"]
    assert average_reward["exclusiveMinimum"] == 0.0
    assert average_reward["exclusiveMaximum"] == 1.0

    episode_reward_items = (
        schema["TaskGradeResponse"]["properties"]["episode_rewards"]["items"]
    )
    assert episode_reward_items["exclusiveMinimum"] == 0.0
    assert episode_reward_items["exclusiveMaximum"] == 1.0
