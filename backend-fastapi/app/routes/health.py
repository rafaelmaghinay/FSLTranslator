"""Health check and general endpoints."""

from fastapi import APIRouter

from app.schemas.responses import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Check backend server availability and status.
    
    Returns:
        Health status response
    """
    return HealthResponse(ok=True, message="Backend is running")
