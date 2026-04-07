#!/usr/bin/env python3
"""Start the warehouse environment API server."""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run
from app.main import app
import uvicorn

if __name__ == "__main__":
    print(f"Starting Warehouse Inventory Environment API")
    print(f"API will be available at http://0.0.0.0:8000")
    print(f"FastAPI docs at http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
