"""Export trained sklearn/TensorFlow artifacts and NOAA history for the static web app."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from src.preprocess import FEATURES, ROOT, prepare_weather_data

PUBLIC = ROOT / "public"


def rounded(value):
    return np.asarray(value, dtype=float).round(8).tolist()


def main():
    PUBLIC.mkdir(exist_ok=True)
    (PUBLIC / "data").mkdir(exist_ok=True)
    (PUBLIC / "models").mkdir(exist_ok=True)
    frame = prepare_weather_data()
    rows = [
        {"date": date.strftime("%Y-%m-%d"), "tmin": round(float(row.tmin), 2),
         "tmax": round(float(row.tmax), 2), "prcp": round(float(row.prcp), 2)}
        for date, row in frame[["tmin", "tmax", "prcp"]].iterrows()
    ]
    (PUBLIC / "data" / "weather.json").write_text(json.dumps({
        "station": "ASN00086282", "startYear": int(frame.index.year.min()),
        "endYear": int(frame.index.year.max()), "records": rows,
    }, separators=(",", ":")))

    regression = joblib.load(ROOT / "models" / "max_temperature_regression.pkl")
    kmeans_artifact = joblib.load(ROOT / "models" / "weather_kmeans.pkl")
    kmeans = kmeans_artifact["pipeline"].named_steps["kmeans"]
    kmeans_scaler = kmeans_artifact["pipeline"].named_steps["standardscaler"]
    assignments = kmeans.predict(kmeans_scaler.transform(frame[kmeans_artifact["features"]]))
    profiles = frame.assign(cluster=assignments).groupby("cluster")[["tmin", "tmax", "prcp"]].mean()

    lstm = keras.models.load_model(ROOT / "models" / "precipitation_lstm.keras")
    lstm_layers = [layer for layer in lstm.layers if isinstance(layer, keras.layers.LSTM)]
    dense_layers = [layer for layer in lstm.layers if isinstance(layer, keras.layers.Dense)]
    scalers = joblib.load(ROOT / "models" / "precipitation_scalers.pkl")
    x_scaler, y_scaler = scalers["x_scaler"], scalers["y_scaler"]
    groups = []
    for i, center in enumerate(kmeans.cluster_centers_):
        groups.append({"id": i, "center": rounded(center),
                       "profile": {key: round(float(profiles.loc[i, key]), 2) for key in ["tmin", "tmax", "prcp"]}})
    groups.sort(key=lambda item: item["profile"]["tmax"])
    for item, name in zip(groups, ["Cooler days", "Mild days", "Warm days", "Hotter days"]):
        item["label"] = name

    model_json = {
        "features": FEATURES,
        "lookback": int(scalers["lookback"]),
        "featureMean": rounded(x_scaler.mean_),
        "featureScale": rounded(x_scaler.scale_),
        "targetScale": rounded(y_scaler.scale_),
        "targetMin": rounded(y_scaler.min_),
        "lstm": [
            {"inputSize": int(layer.input.shape[-1]), "units": int(layer.units),
             "kernel": rounded(layer.get_weights()[0]),
             "recurrent": rounded(layer.get_weights()[1]),
             "bias": rounded(layer.get_weights()[2])}
            for layer in lstm_layers
        ],
        "dense": [
            {"kernel": rounded(layer.get_weights()[0]), "bias": rounded(layer.get_weights()[1]),
             "activation": layer.activation.__name__}
            for layer in dense_layers
        ],
        "regression": {"coefficients": rounded(regression.coef_), "intercept": float(regression.intercept_)},
        "clusters": {"mean": rounded(kmeans_scaler.mean_), "scale": rounded(kmeans_scaler.scale_), "groups": groups},
    }
    (PUBLIC / "models" / "inference.json").write_text(json.dumps(model_json, separators=(",", ":")))
    print(f"Exported {len(rows):,} weather records and browser inference weights to public/.")


if __name__ == "__main__":
    main()
