import joblib

scaler = joblib.load('scaler.pkl')
if hasattr(scaler, 'feature_names_in_'):
    print(f"Features: {list(scaler.feature_names_in_)}")
else:
    print("Features: Unknown")
