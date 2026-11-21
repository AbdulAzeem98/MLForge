"""Application-scoped prediction service for MLForge API.

Manages model lifecycle, artifact loading during application startup, feature
engineering (including pdays sentinel handling), column alignment against the
fitted preprocessor, model dispatch, validation threshold application, and
inference timing.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
import torch

from src.api.schemas import (
    CustomerFeatures,
    HealthResponse,
    MetadataResponse,
    ModelDetail,
    ModelName,
    ModelsResponse,
    PredictionResponse,
)
from src.features.preprocessing import load_fitted_preprocessor
from src.models.train_pytorch import load_pytorch_checkpoint

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PredictionService:
    """Application-scoped service managing preprocessor and model inference."""

    def __init__(
        self,
        preprocessor: Optional[Any] = None,
        lr_model: Optional[Any] = None,
        rf_model: Optional[Any] = None,
        pytorch_model: Optional[Any] = None,
        thresholds: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.preprocessor = preprocessor
        self.lr_model = lr_model
        self.rf_model = rf_model
        self.pytorch_model = pytorch_model
        self.thresholds = thresholds or {}
        self.metadata = metadata or {}

    @classmethod
    def from_artifacts(
        cls,
        project_root: Optional[Path] = None,
    ) -> "PredictionService":
        """Instantiate and populate PredictionService from verified project artifacts.

        Loads:
            1. Preprocessor pipeline (models/preprocessor.joblib)
            2. Logistic Regression (models/logistic_regression.joblib)
            3. Random Forest (models/random_forest.joblib)
            4. PyTorch MLP (models/pytorch_model_best.pt via load_pytorch_checkpoint)
            5. Validation thresholds (experiments/baselines/model_comparison.json)
            6. Pipeline metadata (data/processed/preprocessing_metadata.json)

        Returns:
            Configured PredictionService instance.
        """
        root = Path(project_root or PROJECT_ROOT).resolve()
        logger.info(f"Initializing PredictionService from project root: {root}")

        preprocessor = None
        lr_model = None
        rf_model = None
        pytorch_model = None
        thresholds: Dict[str, float] = {}
        metadata: Dict[str, Any] = {}

        try:
            # 1. Preprocessor
            prep_path = root / "models/preprocessor.joblib"
            if not prep_path.exists():
                raise FileNotFoundError(f"Fitted preprocessor artifact not found at {prep_path}")
            preprocessor = load_fitted_preprocessor(prep_path)
            logger.info("Fitted ColumnTransformer successfully loaded.")

            # 2. Logistic Regression
            lr_path = root / "models/logistic_regression.joblib"
            if not lr_path.exists():
                raise FileNotFoundError(f"Logistic Regression artifact not found at {lr_path}")
            lr_model = joblib.load(lr_path)
            logger.info("Logistic Regression baseline successfully loaded.")

            # 3. Random Forest
            rf_path = root / "models/random_forest.joblib"
            if not rf_path.exists():
                raise FileNotFoundError(f"Random Forest artifact not found at {rf_path}")
            rf_model = joblib.load(rf_path)
            logger.info("Random Forest baseline successfully loaded.")

            # 4. PyTorch Neural Network (reusing load_pytorch_checkpoint)
            pytorch_path = root / "models/pytorch_model_best.pt"
            if not pytorch_path.exists():
                raise FileNotFoundError(f"PyTorch checkpoint artifact not found at {pytorch_path}")
            pytorch_model, _ = load_pytorch_checkpoint(pytorch_path)
            pytorch_model.eval()
            logger.info("PyTorch Neural Network baseline successfully loaded and set to eval mode.")

            # 5. Validation Thresholds from experiments/baselines/model_comparison.json
            baselines_json = root / "experiments/baselines/model_comparison.json"
            if not baselines_json.exists():
                raise FileNotFoundError(f"Validation comparison metrics not found at {baselines_json}")
            with open(baselines_json, "r", encoding="utf-8") as f:
                comp_data = json.load(f)

            name_mapping = {
                "Logistic Regression": "logistic_regression",
                "Random Forest": "random_forest",
                "PyTorch Neural Network": "pytorch",
            }
            for entry in comp_data:
                model_display_name = entry.get("model_name")
                if model_display_name in name_mapping:
                    key = name_mapping[model_display_name]
                    threshold_val = float(entry["tuned_threshold_metrics"]["threshold"])
                    thresholds[key] = threshold_val

            if len(thresholds) != 3:
                raise ValueError(f"Incomplete thresholds extracted from {baselines_json}: {thresholds}")
            logger.info(f"Validation-selected decision thresholds loaded: {thresholds}")

            # 6. Preprocessing Metadata
            prep_meta_path = root / "data/processed/preprocessing_metadata.json"
            if prep_meta_path.exists():
                with open(prep_meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            logger.info("Preprocessing metadata loaded.")

        except Exception as exc:
            # Safeguard 4 & 5: Log detailed internal exceptions and ensure service is marked unready
            logger.exception(f"Failed to initialize PredictionService artifacts: {exc}")
            return cls(
                preprocessor=None,
                lr_model=None,
                rf_model=None,
                pytorch_model=None,
                thresholds={},
                metadata={},
            )

        return cls(
            preprocessor=preprocessor,
            lr_model=lr_model,
            rf_model=rf_model,
            pytorch_model=pytorch_model,
            thresholds=thresholds,
            metadata=metadata,
        )

    def is_ready(self) -> bool:
        """Check whether preprocessor and all three models are ready for inference."""
        return (
            self.preprocessor is not None
            and self.lr_model is not None
            and self.rf_model is not None
            and self.pytorch_model is not None
            and len(self.thresholds) == 3
        )

    def get_health(self) -> HealthResponse:
        """Return readiness and operational health status."""
        ready = self.is_ready()
        return HealthResponse(
            status="healthy" if ready else "unhealthy",
            preprocessor_loaded=self.preprocessor is not None,
            models_loaded={
                "logistic_regression": self.lr_model is not None,
                "random_forest": self.rf_model is not None,
                "pytorch": self.pytorch_model is not None,
            },
            device="cpu",
        )

    def get_models(self) -> ModelsResponse:
        """Return catalog of registered models, architectures, and thresholds."""
        return ModelsResponse(
            default_model=ModelName.RANDOM_FOREST.value,
            models=[
                ModelDetail(
                    model_name="logistic_regression",
                    type="scikit-learn LogisticRegression (L-BFGS, balanced)",
                    threshold=self.thresholds.get("logistic_regression", 0.660),
                    is_loaded=self.lr_model is not None,
                ),
                ModelDetail(
                    model_name="random_forest",
                    type="scikit-learn RandomForestClassifier (100 trees, max_depth=12, balanced)",
                    threshold=self.thresholds.get("random_forest", 0.635),
                    is_loaded=self.rf_model is not None,
                ),
                ModelDetail(
                    model_name="pytorch",
                    type="PyTorch MLForgeClassifier (51 -> 128 -> 64 -> 1, Dropout=0.3)",
                    threshold=self.thresholds.get("pytorch", 0.685),
                    is_loaded=self.pytorch_model is not None,
                ),
            ],
        )

    def get_metadata(self) -> MetadataResponse:
        """Return verified pipeline and platform metadata."""
        return MetadataResponse(
            project="MLForge",
            dataset="UCI Bank Marketing",
            target_column=self.metadata.get("target_column", "y"),
            target_mapping=self.metadata.get("target_mapping", {"no": 0, "yes": 1}),
            excluded_leakage_features=self.metadata.get("excluded_features", ["duration"]),
            raw_input_feature_count=15,
            preprocessor_input_feature_count=16,
            transformed_feature_count=self.metadata.get("transformed_feature_count", 51),
            preprocessing_pipeline={
                "numerical_scaler": "StandardScaler",
                "numerical_imputer": "SimpleImputer(strategy='median')",
                "categorical_encoder": "OneHotEncoder(handle_unknown='ignore')",
                "categorical_imputer": "SimpleImputer(strategy='constant', fill_value='missing')",
            },
            registered_models=[
                "logistic_regression",
                "random_forest",
                "pytorch",
            ],
            validation_thresholds=self.thresholds,
        )

    def preprocess(self, raw_input: Dict[str, Any]) -> np.ndarray:
        """Transform raw customer input into a 51-dimensional feature matrix.

        Applies:
            1. pdays feature engineering:
               - is_previously_contacted = 0 if pdays == -1 else 1
               - pdays_recency = 0.0 if pdays == -1 else float(pdays)
            2. Dropping raw 'pdays' to avoid collinearity.
            3. Verifying and reordering the 16 columns to match prep.feature_names_in_ exactly.
            4. Applying fitted ColumnTransformer.

        Args:
            raw_input: Dictionary containing the 15 raw customer attributes.

        Returns:
            Dense 2D NumPy array with shape (1, 51).
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor is not loaded.")

        df = pd.DataFrame([raw_input])

        # pdays sentinel feature engineering
        pdays_val = df["pdays"].iloc[0]
        df["is_previously_contacted"] = 0 if pdays_val == -1 else 1
        df["pdays_recency"] = 0.0 if pdays_val == -1 else float(pdays_val)
        df = df.drop(columns=["pdays"])

        # Safeguard 2: Verify exact 16-column DataFrame order against preprocessor's feature_names_in_
        expected_cols = list(self.preprocessor.feature_names_in_)
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required preprocessor features: {missing_cols}")

        df_ordered = df[expected_cols]

        transformed_features = self.preprocessor.transform(df_ordered)
        if hasattr(transformed_features, "toarray"):
            transformed_features = transformed_features.toarray()

        feature_names_out = [str(col) for col in self.preprocessor.get_feature_names_out()]
        return pd.DataFrame(transformed_features, columns=feature_names_out)

    def predict(
        self,
        features: CustomerFeatures,
        model_name: ModelName = ModelName.RANDOM_FOREST,
    ) -> PredictionResponse:
        """Execute end-to-end inference and decision thresholding.

        Args:
            features: Validated raw customer features.
            model_name: Selected model architecture.

        Returns:
            PredictionResponse containing probability, threshold, prediction, and latency.
        """
        if not self.is_ready():
            raise RuntimeError("Model service is not ready. Required model artifacts are unavailable.")

        t_start = time.perf_counter()

        raw_dict = features.model_dump()
        X_trans = self.preprocess(raw_dict)

        m_key = model_name.value if isinstance(model_name, ModelName) else str(model_name)
        if m_key not in self.thresholds:
            raise ValueError(f"Unsupported model '{m_key}'. Choose from: {list(self.thresholds.keys())}")

        threshold = self.thresholds[m_key]

        # Model dispatch
        if m_key == ModelName.LOGISTIC_REGRESSION.value:
            probs = self.lr_model.predict_proba(X_trans)
            prob = float(probs[0, 1])

        elif m_key == ModelName.RANDOM_FOREST.value:
            probs = self.rf_model.predict_proba(X_trans.values)
            prob = float(probs[0, 1])

        elif m_key == ModelName.PYTORCH.value:
            self.pytorch_model.eval()
            with torch.no_grad():
                tensor_input = torch.tensor(X_trans.values, dtype=torch.float32)
                logits = self.pytorch_model(tensor_input).squeeze(-1)
                prob_tensor = torch.sigmoid(logits)
                prob = float(prob_tensor.item())

        else:
            raise ValueError(f"Unrecognized model: {m_key}")

        # Clamping probability to [0.0, 1.0]
        prob = max(0.0, min(1.0, prob))

        # Binary decision rule
        binary_pred = 1 if prob >= threshold else 0
        pred_label = "yes" if binary_pred == 1 else "no"

        # Hardware-independent dynamic latency measurement
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return PredictionResponse(
            model_name=m_key,
            probability=round(prob, 4),
            threshold=threshold,
            prediction=binary_pred,
            prediction_label=pred_label,
            latency_ms=round(latency_ms, 3),
        )
