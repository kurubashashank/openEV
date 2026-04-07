#!/bin/sh
set -e

echo "Starting Warehouse Inventory Environment..."
echo "Python version:"
python --version
echo ""

# Check for PORT environment variable (used by HF Spaces)
PORT=${PORT:-7860}
echo "Using port: $PORT"
echo ""

echo "Checking imports..."
python -c "from app.models import State; print('✓ Models imported successfully')"
python -c "from app.environment import WarehouseEnvironment; print('✓ Environment imported successfully')"
python -c "from app.graders import TaskGrader; print('✓ Graders imported successfully')"
python -c "from app.main import app; print('✓ App imported successfully')"

echo ""
echo "Starting Uvicorn server on 0.0.0.0:$PORT..."
exec python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --log-level info
