
import sys
import os

print("Checking imports...")
try:
    import streamlit
    import pandas
    import joblib
    import sklearn
    import fastapi
    import uvicorn
    import structlog
    print("Imports Successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

print("Checking model files...")
try:
    model_path = os.path.join("model", "linear_model.pkl")
    scaler_path = os.path.join("model", "scaler.pkl")
    feature_order_path = os.path.join("model", "feature_order.pkl")
    
    if not os.path.exists(model_path):
        print(f"Model file missing: {model_path}")
        sys.exit(1)
    if not os.path.exists(scaler_path):
        print(f"Scaler file missing: {scaler_path}")
        sys.exit(1)
    if not os.path.exists(feature_order_path):
        print(f"Feature order file missing: {feature_order_path}")
        sys.exit(1)

    print("Model files exist.")
    
    # Try loading
    joblib.load(model_path)
    joblib.load(scaler_path)
    joblib.load(feature_order_path)
    print("Model load successful.")

except Exception as e:
    print(f"Model check failed: {e}")
    sys.exit(1)

print("Project is runnable.")
