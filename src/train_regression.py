"""Train a chronological Linear Regression baseline for daily maximum temperature."""
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.preprocess import ROOT, prepare_weather_data


def main():
    frame = prepare_weather_data().dropna(subset=["tmin", "prcp", "tmax"])
    split = int(len(frame) * 0.8)
    x, y = frame[["tmin", "prcp"]], frame["tmax"]
    model = LinearRegression().fit(x.iloc[:split], y.iloc[:split])
    predicted = model.predict(x.iloc[split:])
    print(f"Holdout MAE: {mean_absolute_error(y.iloc[split:], predicted):.2f} °C")
    print(f"Holdout RMSE: {mean_squared_error(y.iloc[split:], predicted) ** 0.5:.2f} °C")
    print(f"Holdout R²: {r2_score(y.iloc[split:], predicted):.3f}")
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "max_temperature_regression.pkl")


if __name__ == "__main__":
    main()
