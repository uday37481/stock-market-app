import numpy as np
import pandas as pd

from predict import analyze_stock_history, get_prediction


dates = pd.date_range("2024-01-01", periods=120, freq="B")
close = np.linspace(100, 130, 120) + np.sin(np.arange(120)) * 2

history_df = pd.DataFrame(
    {
        "Open": close - 1,
        "High": close + 1.5,
        "Low": close - 2,
        "Close": close,
        "Volume": np.linspace(1_000_000, 1_500_000, 120),
    },
    index=dates,
)

print("Testing Stock Analysis Pipeline...")
try:
    analysis = analyze_stock_history(history_df)
    print(f"Prediction Result: {analysis['prediction']}")
    print(f"Confidence: {analysis['confidence']:.4f}")
    print(f"Predicted Price: {analysis['predicted_price']:.2f}")

    latest_features = analysis["feature_frame"].tail(1).iloc[0].tolist()
    print(f"Legacy get_prediction(): {get_prediction(latest_features)}")
except Exception as e:
    print(f"Error: {e}")
