"""Health check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return application health status without exposing sensitive environment or system paths."""
    return HealthResponse(
        status="ok",
        app_name="Customer Demand Forecasting & Business Intelligence Platform",
        version="1.0.0"
    )
