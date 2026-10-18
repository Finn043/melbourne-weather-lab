"""Fit a standardized K-Means model for common weather-condition groups."""
import joblib
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.preprocess import ROOT, prepare_weather_data


def main():
    frame = prepare_weather_data().dropna(subset=["tmin", "tmax", "prcp"])
    features = ["tmin", "tmax", "prcp"]
    pipeline = make_pipeline(StandardScaler(), KMeans(n_clusters=4, random_state=42, n_init=10))
    pipeline.fit(frame[features])
    frame["weather_group"] = pipeline.predict(frame[features])
    print("Cluster profiles (mean °C / mm):")
    print(frame.groupby("weather_group")[features].mean().round(2).to_string())
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": features}, ROOT / "models" / "weather_kmeans.pkl")


if __name__ == "__main__":
    main()
