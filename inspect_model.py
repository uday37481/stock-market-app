import joblib

try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('stock_model.pkl')
    
    if hasattr(scaler, 'n_features_in_'):
        print(f"Scaler Features: {scaler.n_features_in_}")
    else:
        print("Scaler Features: Unknown")
        
    if hasattr(model, 'n_features_in_'):
        print(f"Model Features: {model.n_features_in_}")
    else:
        print("Model Features: Unknown")
except Exception as e:
    print(f"Error loading files: {e}")
