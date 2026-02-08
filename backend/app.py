"""
Manufacturing Efficiency Predictor API
Capstone backend: prediction, model info, dataset stats, and structured logging.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODEL_DIR = PROJECT_ROOT / "model"
DATA_CSV = PROJECT_ROOT / "data" / "manufacturing_dataset_1000_samples.csv"
METADATA_JSON = MODEL_DIR / "model_metadata.json"

# Structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()

# Model & dataset state
model = None
scaler = None
feature_order: Optional[list] = None
model_metadata: Optional[dict] = None
dataset_df: Optional[pd.DataFrame] = None
dataset_numeric: Optional[pd.DataFrame] = None


def load_model_artifacts() -> None:
    global model, scaler, feature_order, model_metadata
    try:
        model = joblib.load(MODEL_DIR / "linear_model.pkl")
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        feature_order = joblib.load(MODEL_DIR / "feature_order.pkl")
        logger.info("model_loaded", path=str(MODEL_DIR), features=len(feature_order or []))
    except FileNotFoundError as e:
        logger.warning("model_not_loaded", path=str(MODEL_DIR), error=str(e))
        model = None
        scaler = None
        feature_order = None

    model_metadata = None
    if METADATA_JSON.exists():
        try:
            with open(METADATA_JSON) as f:
                model_metadata = json.load(f)
        except Exception as e:
            logger.warning("metadata_not_loaded", error=str(e))


def load_dataset() -> None:
    global dataset_df, dataset_numeric
    if not DATA_CSV.exists():
        logger.warning("dataset_not_found", path=str(DATA_CSV))
        return
    try:
        dataset_df = pd.read_csv(DATA_CSV)
        dataset_numeric = dataset_df.select_dtypes(include=["int64", "float64"])
        logger.info(
            "dataset_loaded",
            path=str(DATA_CSV),
            rows=len(dataset_df),
            numeric_columns=len(dataset_numeric.columns),
        )
    except Exception as e:
        logger.warning("dataset_load_failed", path=str(DATA_CSV), error=str(e))
        dataset_df = None
        dataset_numeric = None


# FastAPI app
app = FastAPI(
    title="Manufacturing Efficiency Predictor API",
    description="Predict Parts Per Hour from machine and process parameters.",
    version="1.0.0",
)


@app.on_event("startup")
def startup():
    load_model_artifacts()
    load_dataset()


# Request/response models
class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Feature names to numeric values")

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "Injection_Temperature": 220.0,
                    "Injection_Pressure": 130.0,
                    "Cycle_Time": 28.0,
                    "Cooling_Time": 14.0,
                    "Material_Viscosity": 300.0,
                    "Ambient_Temperature": 26.0,
                    "Machine_Age": 5.0,
                    "Operator_Experience": 10.0,
                    "Maintenance_Hours": 50,
                    "Temperature_Pressure_Ratio": 1.7,
                    "Total_Cycle_Time": 42.0,
                    "Efficiency_Score": 0.06,
                    "Machine_Utilization": 0.5,
                }
            }
        }


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted Parts Per Hour")
    status: str = Field(default="success", description="Status of the prediction")


class ModelInfoResponse(BaseModel):
    model_type: str = Field(..., description="e.g. LinearRegression")
    feature_count: int = Field(..., description="Number of input features")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    training_date: Optional[str] = None
    version: Optional[str] = None
    metrics: Optional[dict] = None


# Middleware: request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = datetime.utcnow()
    response = await call_next(request)
    duration_ms = (datetime.utcnow() - start).total_seconds() * 1000
    log_payload = {
        "event": "request",
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": round(duration_ms, 2),
    }
    logger.info(**log_payload)
    return response


# Health & model info
@app.get("/", tags=["Health"])
def root():
    """Health check and readiness (model loaded)."""
    return {
        "status": "Backend is running",
        "model_loaded": model is not None and scaler is not None and feature_order is not None,
    }


@app.get("/health", tags=["Health"])
def health():
    """Same as GET / for explicit health checks."""
    return root()


@app.get("/features", tags=["Prediction"])
def get_features():
    """Return the list of required feature names for POST /predict."""
    if feature_order is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"features": feature_order}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Return model metadata: type, feature count, optional training date/version/metrics."""
    model_type = "LinearRegression"
    if model is not None:
        model_type = type(model).__name__
    return ModelInfoResponse(
        model_type=model_type,
        feature_count=len(feature_order) if feature_order else 0,
        model_loaded=model is not None and scaler is not None and feature_order is not None,
        training_date=model_metadata.get("training_date") if model_metadata else None,
        version=model_metadata.get("version") if model_metadata else None,
        metrics=model_metadata.get("metrics") if model_metadata else None,
    )


# Prediction
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(body: PredictionRequest):
    """Accept feature values (JSON), return predicted Parts Per Hour."""
    if model is None or scaler is None or feature_order is None:
        raise HTTPException(
            status_code=503,
            detail="Model files not loaded. Run training and ensure model/ exists.",
        )

    missing = set(feature_order) - set(body.features.keys())
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required features: {sorted(missing)}")

    input_row = {f: body.features.get(f, 0.0) for f in feature_order}
    input_df = pd.DataFrame([input_row])

    try:
        X = scaler.transform(input_df)
        pred = float(model.predict(X)[0])
    except Exception as e:
        logger.exception("predict_error", error=str(e), features_keys=list(body.features.keys()))
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    logger.info(
        "prediction",
        path="/predict",
        prediction=pred,
        input_keys=list(body.features.keys()),
    )
    return PredictionResponse(prediction=pred, status="success")


# Dataset stats & sample
@app.get("/stats", tags=["Dataset"])
@app.get("/dataset/summary", tags=["Dataset"])
def dataset_summary():
    """Summary statistics (mean, std, min, max) for numeric columns in the dataset."""
    if dataset_numeric is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded or file missing.")
    stats = dataset_numeric.describe().round(4).to_dict()
    return {
        "summary": stats,
        "rows": int(len(dataset_numeric)),
        "columns": list(dataset_numeric.columns),
    }


@app.get("/dataset/sample", tags=["Dataset"])
def dataset_sample(n: int = 1):
    """Return one or a few sample rows (numeric only) for prefilling the form."""
    if dataset_numeric is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded or file missing.")
    n = max(1, min(n, 10))
    sample = (
        dataset_numeric.drop(columns=["Parts_Per_Hour"], errors="ignore")
        .head(100)
        .sample(n=n, random_state=42)
    )
    return {"sample": sample.to_dict(orient="records")}


# Global exception handler for consistent JSON errors
@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code},
    )
