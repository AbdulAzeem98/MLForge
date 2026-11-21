"""FastAPI entry point for MLForge Model Serving.

Provides RESTful endpoints for health checks, model catalog inspection,
pipeline metadata, and threshold-optimized inference across all three baseline models.
Uses application-scoped lifecycle management via FastAPI lifespan and dependency injection.
"""

from contextlib import asynccontextmanager
import logging
from typing import Any, Dict
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.schemas import (
    HealthResponse,
    MetadataResponse,
    ModelsResponse,
    PredictRequest,
    PredictionResponse,
)
from src.api.service import PredictionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("src.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application-scoped lifecycle manager for loading and unloading model artifacts."""
    existing_service = getattr(app.state, "prediction_service", None)
    if existing_service is not None:
        logger.info("PredictionService already provided in application state.")
        service = existing_service
    else:
        logger.info("Initializing MLForge PredictionService during application startup...")
        service = PredictionService.from_artifacts()
        app.state.prediction_service = service

    if service is None or not service.is_ready():
        logger.error("WARNING: PredictionService failed to load one or more required artifacts on startup.")
    else:
        logger.info("MLForge PredictionService initialized successfully with all 3 models and preprocessor.")

    yield

    logger.info("Shutting down MLForge PredictionService...")
    app.state.prediction_service = None


app = FastAPI(
    title="MLForge Model Serving API",
    description="Production REST API serving Bank Marketing term deposit classification baselines.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_prediction_service(request: Request) -> PredictionService:
    """FastAPI dependency injecting the application-scoped PredictionService."""
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        logger.error("PredictionService requested but not found in application state.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently unavailable.",
        )
    return service


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler ensuring detailed internal logging and sanitized client responses."""
    logger.exception(f"Unhandled server error on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred during request processing."},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and readiness probe",
    tags=["Diagnostics"],
)
async def health_check(
    response: Response,
    service: PredictionService = Depends(get_prediction_service),
) -> HealthResponse:
    """Check whether the preprocessor and all baseline models are loaded and ready."""
    health = service.get_health()
    if health.status != "healthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        response.status_code = status.HTTP_200_OK
    return health


@app.get(
    "/models",
    response_model=ModelsResponse,
    summary="List available prediction models",
    tags=["Metadata"],
)
async def list_models(
    service: PredictionService = Depends(get_prediction_service),
) -> ModelsResponse:
    """Retrieve catalog of available models, architectures, and validation decision thresholds."""
    if not service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model catalogue is unavailable because service failed initialization.",
        )
    return service.get_models()


@app.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Pipeline and dataset metadata",
    tags=["Metadata"],
)
async def get_metadata(
    service: PredictionService = Depends(get_prediction_service),
) -> MetadataResponse:
    """Retrieve verified feature engineering and preprocessing pipeline specifications."""
    if not service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline metadata is unavailable because service failed initialization.",
        )
    return service.get_metadata()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict customer term deposit subscription",
    tags=["Inference"],
)
async def predict_deposit(
    request: PredictRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """Perform customer scoring using selected model and validation-tuned decision threshold."""
    if not service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is unavailable: models or preprocessor not loaded.",
        )

    try:
        prediction_result = service.predict(
            features=request.features,
            model_name=request.model_name,
        )
        return prediction_result
    except ValueError as val_err:
        logger.warning(f"Validation or model error during inference: {val_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.exception(f"Inference computation failure: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference execution failed during computation.",
        )
