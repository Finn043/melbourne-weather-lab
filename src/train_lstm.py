"""Train an LSTM to forecast next-day precipitation from recent daily weather."""
from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from src.preprocess import FEATURES, ROOT, make_sequences, prepare_weather_data

LOOKBACK = 30
MODEL_DIR = ROOT / "models"


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    frame = prepare_weather_data()
    split = int(len(frame) * 0.8)
    if split <= LOOKBACK or len(frame) - split < 30:
        raise ValueError("Need at least 180 daily observations to train and evaluate the LSTM.")
    x_raw, y_raw = make_sequences(frame, LOOKBACK)
    train_sequence_count = split - LOOKBACK
    x_scaler, y_scaler = StandardScaler(), MinMaxScaler()
    x_scaler.fit(frame[FEATURES].iloc[:split])
    x_scaled = x_scaler.transform(frame[FEATURES]).astype("float32")
    x = np.stack([x_scaled[i - LOOKBACK:i] for i in range(LOOKBACK, len(frame))])
    # log1p reduces the impact of rare heavy-rain days while retaining millimetre output.
    transformed_rain = np.log1p(y_raw).reshape(-1, 1)
    y_scaler.fit(transformed_rain[:train_sequence_count])
    y = y_scaler.transform(transformed_rain).astype("float32").ravel()
    model = keras.Sequential([
        keras.layers.Input(shape=(LOOKBACK, len(FEATURES))),
        keras.layers.LSTM(48, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(24),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
    model.fit(x[:train_sequence_count], y[:train_sequence_count], validation_split=0.15,
              epochs=80, batch_size=32, callbacks=callbacks, shuffle=False, verbose=1)
    model.save(MODEL_DIR / "precipitation_lstm.keras")
    joblib.dump({"x_scaler": x_scaler, "y_scaler": y_scaler, "lookback": LOOKBACK, "features": FEATURES},
                MODEL_DIR / "precipitation_scalers.pkl")
    print(f"Saved LSTM and preprocessing artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
