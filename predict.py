import numpy as np
import pandas as pd

from model_loader import load_model, load_scaler

FEATURE_NAMES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA_10",
    "MA_50",
    "RSI",
    "MACD",
    "BB_Upper",
    "BB_Lower",
    "Momentum",
    "Volatility",
]


def calculate_rsi(close_series, period=14):
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def sanitize_numeric_frame(df):
    clean_df = df.copy()
    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
    clean_df = clean_df.fillna(0)
    for column in clean_df.columns:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce").fillna(0.0)
    return clean_df


def prepare_features(history_df):
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [col for col in required_columns if col not in history_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    df = sanitize_numeric_frame(history_df[required_columns])
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["RSI"] = calculate_rsi(df["Close"])

    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26

    rolling_std = df["Close"].rolling(window=20).std()
    rolling_mean = df["Close"].rolling(window=20).mean()
    df["BB_Upper"] = rolling_mean + (rolling_std * 2)
    df["BB_Lower"] = rolling_mean - (rolling_std * 2)
    df["Momentum"] = df["Close"] - df["Close"].shift(5)
    df["Volatility"] = df["Close"].pct_change().rolling(window=10).std().fillna(0)

    feature_df = sanitize_numeric_frame(df[FEATURE_NAMES])
    if feature_df.empty:
        raise ValueError("Not enough historical data to compute indicators. Use at least 60 trading days.")

    return feature_df


def _get_prediction_confidence(model, scaled_data, predicted_class):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_data)[0]
        return float(np.max(probabilities))

    if hasattr(model, "decision_function"):
        decision = model.decision_function(scaled_data)
        score = decision[0] if np.ndim(decision) else decision
        confidence = 1 / (1 + np.exp(-abs(score)))
        return float(confidence)

    return 0.5 if predicted_class in [0, 1] else 0.0


def predict_from_features(feature_df):
    scaler = load_scaler()
    model = load_model()

    latest_features = feature_df.tail(1).copy()
    latest_features = latest_features.reindex(columns=FEATURE_NAMES)
    latest_features = sanitize_numeric_frame(latest_features)
    scaled_data = scaler.transform(latest_features)

    predicted_class = int(model.predict(scaled_data)[0])
    prediction = "UP" if predicted_class == 1 else "DOWN"
    confidence = _get_prediction_confidence(model, scaled_data, predicted_class)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "predicted_class": predicted_class,
        "latest_features": latest_features.iloc[0].to_dict(),
    }


def estimate_future_price(current_price, volatility, confidence, prediction):
    safe_current_price = 0.0 if current_price is None or pd.isna(current_price) else float(current_price)
    safe_volatility = 0.0 if volatility is None or pd.isna(volatility) else float(volatility)
    safe_confidence = 0.5 if confidence is None or pd.isna(confidence) else float(confidence)

    base_move = max(safe_volatility, 0.005)
    adjusted_move = min(base_move * (0.8 + safe_confidence), 0.12)
    direction = 1 if prediction == "UP" else -1
    return float(safe_current_price * (1 + (direction * adjusted_move)))


def analyze_stock_history(history_df):
    feature_df = prepare_features(history_df)
    prediction_result = predict_from_features(feature_df)

    current_price = history_df["Close"].iloc[-1]
    recent_volatility = feature_df["Volatility"].iloc[-1]

    current_price = 0.0 if current_price is None or pd.isna(current_price) else float(current_price)
    recent_volatility = 0.0 if recent_volatility is None or pd.isna(recent_volatility) else float(recent_volatility)
    predicted_price = estimate_future_price(
        current_price=current_price,
        volatility=recent_volatility,
        confidence=prediction_result["confidence"],
        prediction=prediction_result["prediction"],
    )

    return {
        **prediction_result,
        "current_price": current_price,
        "predicted_price": predicted_price,
        "feature_frame": feature_df,
    }


def get_prediction(input_data):
    try:
        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        return predict_from_features(input_df)["prediction"]
    except Exception as e:
        return f"Error during prediction: {e}"
