from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint_reports_healthy():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_reset_allows_medium_task_and_sets_active_environment():
    with TestClient(app) as client:
        reset_response = client.post("/reset", json={"task_id": "medium", "seed": 7})
        step_response = client.post(
            "/step",
            json={"action": {"order_quantities": [10, 20, 30]}},
        )

    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["task_id"] == "medium"
    assert len(reset_payload["state"]["inventory_levels"]) == 3

    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert len(step_payload["state"]["inventory_levels"]) == 3
    assert step_payload["state"]["current_step"] == 1


def test_reset_accepts_empty_body_and_defaults_to_easy():
    with TestClient(app) as client:
        response = client.post("/reset")

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == "easy"
    assert len(payload["state"]["inventory_levels"]) == 1


def test_step_requires_reset_before_use():
    with TestClient(app) as client:
        response = client.post("/step", json={"action": {"order_quantities": [10]}})

    assert response.status_code == 400
    assert "Call /reset first" in response.json()["detail"]
