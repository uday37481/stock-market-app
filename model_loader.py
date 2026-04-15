import joblib
import os

def load_scaler(scaler_path='scaler.pkl'):
    """Loads the pre-trained scaler."""
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    else:
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

def load_model(model_path='stock_model.pkl'):
    """Loads the pre-trained stock market model."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
