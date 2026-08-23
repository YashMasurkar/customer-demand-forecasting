"""FastAPI application factory and middleware configuration."""

import os
from pathlib import Path
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.api.routes import health, analytics, forecast, models, analyst


def get_allowed_cors_origins() -> List[str]:
    """Return configured CORS origins from environment or sensible local defaults."""
    env_origins = os.getenv("CORS_ORIGINS", "")
    if env_origins.strip():
        return [orig.strip() for orig in env_origins.split(",") if orig.strip()]
    return [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]


def create_app() -> FastAPI:
    """Create and configure FastAPI application instance."""
    app = FastAPI(
        title="DemandIQ — Customer Demand Forecasting & Business Intelligence Platform",
        description="Grounded business analytics, time-series forecasting, and LLM analyst API.",
        version="1.0.0"
    )

    # Configure secure CORS
    origins = get_allowed_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Register API routes for both /api and /api/v1 prefixes
    for prefix in ["/api", "/api/v1"]:
        app.include_router(health.router, prefix=prefix)
        app.include_router(analytics.router, prefix=prefix)
        app.include_router(forecast.router, prefix=prefix)
        app.include_router(models.router, prefix=prefix)
        app.include_router(analyst.router, prefix=prefix)

    # Mount static assets for frontend dashboard
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        def serve_index():
            index_path = static_dir / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return {"message": "Welcome to DemandIQ API. Visit /docs for Swagger UI."}

    return app


# Default app instance for uvicorn
app = create_app()
