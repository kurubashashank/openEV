"""Server entrypoint expected by OpenEnv validation."""

from app.main import app


def main() -> None:
    """Run the API server via uvicorn."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")


if __name__ == "__main__":
    main()
