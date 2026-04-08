---
title: Warehouse Inventory Environment
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: openrail
---

# Warehouse Inventory Environment

Dockerized FastAPI environment for warehouse inventory control tasks. The API
implements the core OpenEnv loop with `/reset`, `/step`, `/state`, `/grade`,
and `/health` endpoints.

## Tasks

- `easy`: 1 product, 50 steps
- `medium`: 3 products, 100 steps
- `hard`: 5 products, 150 steps

The reward is normalized to the range `[0.0, 1.0]` and penalizes both excess
inventory holding and stockouts.

## Local Run

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## Docker Build

```bash
docker build -t warehouse-env-v2 .
docker run --rm -p 7860:7860 warehouse-env-v2
```

The container uses `start.sh`, listens on `PORT` when provided, and exposes
`/health` for health checks.

## Quick API Check

```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{\"task_id\":\"easy\"}"
```

## Project Files

- `app/main.py`: FastAPI endpoints
- `app/environment.py`: inventory simulator
- `app/graders.py`: baseline grading helpers
- `openenv.yaml`: environment contract
- `inference.py`: baseline interaction script

## Verification

- Docker image builds successfully
- Container health endpoint returns `200 OK`
- API regression tests pass
