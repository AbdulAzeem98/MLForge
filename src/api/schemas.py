"""Pydantic schemas and request/response contracts for MLForge API.

Defines input validation contracts with strict extra-field rejection (forbidding
data leakage attributes such as 'duration' and target 'y'), prediction request
structures, inference response payloads, and metadata schemas.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ModelName(str, Enum):
    """Supported model identifiers for inference."""

    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    PYTORCH = "pytorch"


class CustomerFeatures(BaseModel):
    """Raw customer attributes matching the 15 predictive Bank Marketing features.

    Note:
        Strictly forbids extra fields to immediately reject target leakage ('duration'),
        ground truth labels ('y'), and arbitrary unknown attributes.
    """

    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=18, le=120, description="Customer age in years (18-120)")
    job: str = Field(..., min_length=1, max_length=50, description="Job type (e.g., 'admin.', 'blue-collar', 'management', 'unknown')")
    marital: str = Field(..., min_length=1, max_length=20, description="Marital status ('married', 'single', 'divorced')")
    education: str = Field(..., min_length=1, max_length=20, description="Education level ('primary', 'secondary', 'tertiary', 'unknown')")
    default: str = Field(..., min_length=1, max_length=10, description="Has credit in default? ('no', 'yes')")
    balance: float = Field(..., description="Average yearly balance in euros")
    housing: str = Field(..., min_length=1, max_length=10, description="Has housing loan? ('no', 'yes')")
    loan: str = Field(..., min_length=1, max_length=10, description="Has personal loan? ('no', 'yes')")
    contact: str = Field(..., min_length=1, max_length=20, description="Contact communication type ('cellular', 'telephone', 'unknown')")
    day: int = Field(..., ge=1, le=31, description="Last contact day of the month (1-31)")
    month: str = Field(..., min_length=3, max_length=10, description="Last contact month of year ('jan' through 'dec')")
    campaign: int = Field(..., ge=1, description="Number of contacts performed during this campaign (>= 1)")
    pdays: int = Field(..., ge=-1, description="Number of days passed after client was last contacted (-1 means not previously contacted)")
    previous: int = Field(..., ge=0, description="Number of contacts performed before this campaign (>= 0)")
    poutcome: str = Field(..., min_length=1, max_length=20, description="Outcome of previous marketing campaign ('failure', 'other', 'success', 'unknown')")


class PredictRequest(BaseModel):
    """Prediction request payload wrapping customer features and model selection."""

    model_config = ConfigDict(extra="forbid")

    model_name: Optional[ModelName] = Field(
        default=ModelName.RANDOM_FOREST,
        description="Target model for prediction. Defaults to 'random_forest'.",
    )
    features: CustomerFeatures = Field(
        ...,
        description="Raw customer attributes (15 predictive features).",
    )


class PredictionResponse(BaseModel):
    """Inference response schema returned to clients."""

    model_name: str = Field(..., description="Name of the model used for prediction")
    probability: float = Field(..., ge=0.0, le=1.0, description="Estimated positive-class probability P(y=1)")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Validation-selected decision threshold applied")
    prediction: int = Field(..., description="Binary classification outcome (0 or 1)")
    prediction_label: str = Field(..., description="Human-readable decision label ('no' or 'yes')")
    latency_ms: float = Field(..., ge=0.0, description="Measured inference latency in milliseconds")


class HealthResponse(BaseModel):
    """System health and model readiness status schema."""

    status: str = Field(..., description="Overall service status ('healthy', 'degraded', 'unhealthy')")
    preprocessor_loaded: bool = Field(..., description="Whether the feature preprocessor is loaded")
    models_loaded: Dict[str, bool] = Field(..., description="Status of each individual model")
    device: str = Field(..., description="Inference device ('cpu')")


class ModelDetail(BaseModel):
    """Specification of an individual registered model."""

    model_name: str = Field(..., description="Internal model name")
    type: str = Field(..., description="Architecture / model type description")
    threshold: float = Field(..., description="Validation-tuned F1 decision threshold")
    is_loaded: bool = Field(..., description="Whether the model checkpoint is loaded in memory")


class ModelsResponse(BaseModel):
    """List of available models and default configuration."""

    default_model: str = Field(..., description="Default model when unspecified")
    models: List[ModelDetail] = Field(..., description="Registered models catalogue")


class MetadataResponse(BaseModel):
    """Verified platform, preprocessing, and pipeline metadata."""

    project: str = Field(..., description="Project name")
    dataset: str = Field(..., description="Source dataset")
    target_column: str = Field(..., description="Target variable name")
    target_mapping: Dict[str, int] = Field(..., description="Target class mapping")
    excluded_leakage_features: List[str] = Field(..., description="Features excluded to prevent data leakage")
    raw_input_feature_count: int = Field(..., description="Number of raw input features accepted")
    preprocessor_input_feature_count: int = Field(..., description="Number of columns passed to preprocessor")
    transformed_feature_count: int = Field(..., description="Number of transformed features produced")
    preprocessing_pipeline: Dict[str, str] = Field(..., description="Transformers used in preprocessing")
    registered_models: List[str] = Field(..., description="Registered model identifiers")
    validation_thresholds: Dict[str, float] = Field(..., description="Validation-tuned decision thresholds")
