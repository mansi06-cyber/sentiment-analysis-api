"""Health check endpoint router."""
from fastapi import APIRouter, status
from app.schemas import HealthResponse
from app.services import sentiment_service

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check API and Model Health",
    description="Returns operational status of the service, model loading state, and compute device."
)
async def get_health() -> HealthResponse:
    """Check API and model health status."""
    is_ready = sentiment_service.is_loaded
    return HealthResponse(
        status="healthy" if is_ready else "loading",
        model_loaded=is_ready,
        model_name=sentiment_service.model_name,
        device=sentiment_service.device
    )
