# Melbourne Weather Lab

An interactive weather analysis dashboard for **Melbourne Airport, Victoria**. It combines daily NOAA observations with three small machine learning models and runs prediction directly in the browser.

The project is designed to demonstrate a complete data and product workflow: collecting and cleaning real data, exploring it, training models, exporting models for browser inference, building a typed responsive frontend, and deploying a static application to Netlify.

> The predictions are educational estimates from historical observations. They are not official forecasts, calibrated probabilities, or severe-weather advice.

## Live application

**Netlify URL:** <https://melbourne-weather-lab-finn043.netlify.app>

This repository is connected to that Netlify site. Pushes to `main` trigger production deploys automatically.

## What the application does

- Shows the latest available daily maximum temperature, minimum temperature, and precipitation from the station record.
- Forecasts next-day precipitation with a two-layer LSTM using the previous 30 observed days.
- Estimates maximum temperature from user-selected minimum temperature and precipitation with Linear Regression.
- Assigns a set of weather conditions to the closest of four K-Means groups.
- Plots daily temperatures and rainfall across a selectable 90-day, 1-year, 5-year, or full-record view.
- Runs all inference in the browser. The deployed app has no Python API server and needs no runtime API key.

## Technology

| Area | Technology | What it does here |
|---|---|---|
| Frontend | React 18 | Dashboard components and interactive state |
| Language | TypeScript, strict mode | Typed data contracts and browser-side inference |
| Styling | Tailwind CSS 4 + project CSS | Responsive layout, utility styling, charts and visual states |
| Build tool | Vite 6 | Local development server and static production build |
| Icons | Lucide React | Consistent interface icons |
| Data processing | Python, pandas, NumPy | NOAA parsing, daily index, imputation and features |
| Model training | scikit-learn | Linear Regression, scaling and K-Means |
| Sequence model | TensorFlow / Keras | LSTM model training and weight export |
| Model artifacts | JSON in `public/` | Weather history and trained weights served as static assets |
| Hosting | Netlify | Builds and hosts the static Vite output from `dist/` |
| Exploration | Jupyter Notebook, Matplotlib | Initial time-series and seasonal analysis |

## Architecture

```text
NOAA GHCN-Daily (.dly)
        │
        ▼
src/preprocess.py ──► daily clean data + calendar features
        │                         │
        │                         ├──► src/train_regression.py ──► Linear Regression
        │                         ├──► src/train_kmeans.py ──────► K-Means
        │                         └──► src/train_lstm.py ────────► Keras LSTM
        │                                                       │
        └───────────────────────────────────────────────────────┤
                                                                ▼
                                              src/export_web_artifacts.py
                                                ├── public/data/weather.json
                                                └── public/models/inference.json
                                                                │
                                                                ▼
                                      React + TypeScript + Tailwind (src-web/)
                                      Browser-side TypeScript inference
                                                                │
                                                                ▼
                                                       Netlify static hosting
```

Training uses Python because the preprocessing and ML ecosystem is mature and the supplied project specification calls for scikit-learn and Keras. A trained model should not require TensorFlow in every visitor's browser or a continuously running Python server. The export script serializes the model weights and scalers as JSON; small TypeScript functions apply those parameters client-side. This keeps the hosted app static, inexpensive to run, and easy to deploy.

## Data source and preparation

The project uses NOAA NCEI's [GHCN-Daily record](https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/ASN00086282.dly) for Melbourne Airport, station ID **ASN00086282**. NOAA station metadata is in the [GHCN-Daily station list](https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt).

The source is a fixed-width `.dly` file. `src/preprocess.py` reads TMAX, TMIN, and PRCP fields, converts the stored tenths into degrees Celsius and millimetres, pivots measurements into daily columns, and sorts the dates. Preprocessing then:

1. Builds a continuous daily date index.
2. Treats missing precipitation as `0 mm`.
3. Interpolates temperature gaps up to seven days and fills remaining leading/trailing gaps from the nearest observation.
4. Adds day-of-year and long-term position as sine/cosine cyclic features for the LSTM.
5. Writes `data/melbourne_weather.csv` for training and exploration.

The browser build includes a compact JSON copy of the processed daily observations. The raw download and Python model files are ignored by Git; the exported files under `public/` are intentional deployment assets.

## Machine learning models

### 1. Next-day precipitation — LSTM regression

**Input:** 30 consecutive days of seven scaled features: minimum temperature, maximum temperature, precipitation, two annual-cycle values, and two long-term-cycle values.

**Output:** estimated precipitation amount in millimetres for the next day. The target is transformed with `log1p` during training to reduce the influence of rare heavy-rain days, then returned to millimetres with `expm1` in the app. It predicts an amount, not a probability of rain.

The network has two LSTM layers (48 and 24 units), dropout during training, a 16-unit ReLU dense layer, and a linear output. Data is kept in date order. The first 80% of the date range is used for training; early stopping uses a validation subset from that training period. The final chronological 20% is held out from fitting, although this initial project does not report a separate LSTM holdout score.

### 2. Maximum temperature — Linear Regression

**Inputs:** minimum temperature (°C) and daily precipitation (mm).

**Output:** estimated maximum temperature (°C).

This is an interpretable baseline: the model computes a weighted sum of the two inputs plus an intercept. A chronological 80/20 split is used; the script prints MAE, RMSE, and R² on the final 20%. On the NOAA snapshot used for the current model, the holdout results were approximately **MAE 3.67 °C, RMSE 4.69 °C, R² 0.488**. This is a useful baseline, not an operational forecasting result.

### 3. Weather patterns — K-Means clustering

**Inputs:** minimum temperature, maximum temperature, and precipitation. Each feature is standardized before distance calculations so millimetres and degrees do not dominate merely because of their units. K-Means uses four clusters and a fixed random seed for reproducibility.

The UI assigns readable names in order of cluster-average maximum temperature. These names are presentation labels only; cluster IDs have no natural order and are not official categories. The table shows each cluster's mean conditions so the user can interpret it.

## Local setup

### Requirements

- Python **3.10–3.12** for the data and model workflow. TensorFlow wheel support depends on OS and processor architecture.
- Node.js **22** (Node 20+ should work locally) and npm.
- Internet access for the first NOAA download and Google Fonts. The checked-in static JSON allows the frontend itself to run without calling NOAA at runtime.

### 1. Install frontend packages

```bash
npm install
```

### 2. Install Python packages (only needed to download, retrain, or export data/models)

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the app locally

```bash
npm run dev
```

Open the local URL printed by Vite. The frontend loads `public/data/weather.json` and `public/models/inference.json` and runs all predictions in TypeScript.

### 4. Build and preview production output

```bash
npm run build
npm run preview
```

The production site is written to `dist/`. The same command is configured in `netlify.toml`.

## Refresh NOAA data and retrain

The first setup can download and prepare the station file with:

```bash
python -c "from src.preprocess import prepare_weather_data; print(prepare_weather_data().tail())"
```

To refresh an existing data file from NOAA, retrain the models, and regenerate the static browser assets:

```bash
python -c "from src.preprocess import prepare_weather_data; prepare_weather_data(force_download=True)"
python -m src.train_regression
python -m src.train_kmeans
python -m src.train_lstm
python -m src.export_web_artifacts
npm run build
```

The corresponding npm shortcuts are `npm run train:regression`, `npm run train:kmeans`, `npm run train:lstm`, and `npm run export:model`. Run those after activating the Python environment and installing `requirements.txt`.

The training scripts write artifacts into `models/`, which is gitignored. The export script reads those artifacts and regenerates `public/data/weather.json` and `public/models/inference.json`; commit these two public files if the refreshed model should be included in the next Netlify deploy.

## Deploy to Netlify

This repository includes `netlify.toml` with the Vite build command and `dist` publish directory.

### Connect a Git repository (recommended)

1. Push this project to GitHub or another Git provider.
2. In Netlify choose **Add new site → Import an existing project** and authorize the repository.
3. Confirm the build command is `npm run build` and publish directory is `dist` (already defined in `netlify.toml`).
4. Deploy. Subsequent pushes trigger a new build automatically.

No environment variables are required. The dataset and model are static public files. If you update the NOAA snapshot or train new models, commit the regenerated files under `public/` and push to trigger a deploy.

### Deploy with the Netlify CLI

```bash
npm install
npm run build
npx netlify login
npx netlify init          # link an existing site or create a new one
npx netlify deploy --prod --dir=dist
```

The CLI can also create a new site if you have not created one yet. Do not upload `.env`, source credentials, raw personal data, or private model artifacts to `public/`.

### Netlify settings

- Build command: `npm run build`
- Publish directory: `dist`
- Node version: `22` (set in `netlify.toml`)
- Functions: none
- Required environment variables: none

## Project structure

```text
.
├── index.html                    # Vite HTML shell and page metadata
├── netlify.toml                  # Netlify build, publish and asset caching config
├── package.json                  # Frontend scripts and dependencies
├── package-lock.json             # Locked npm dependency versions
├── public/
│   ├── data/weather.json         # Processed public station history for the UI
│   └── models/inference.json     # Exported model weights, scalers and centroids
├── src-web/
│   ├── App.tsx                   # Dashboard views and interactive controls
│   ├── inference.ts              # TypeScript implementations for browser inference
│   ├── main.tsx                  # React entry point
│   ├── styles.css                # Tailwind entry and dashboard styles
│   └── types.ts                  # Shared data/model types
├── src/
│   ├── preprocess.py             # NOAA download, parser, missing values, features
│   ├── train_lstm.py             # Precipitation sequence model
│   ├── train_regression.py       # Maximum temperature regression
│   ├── train_kmeans.py           # Weather condition clusters
│   └── export_web_artifacts.py   # Serialize data and fitted models to public JSON
├── data/                         # Downloaded NOAA files and local processed CSV (ignored)
├── models/                       # Local Keras and sklearn training artifacts (ignored)
├── notebooks/                    # EDA notebook
├── requirements.txt              # Python data/model workflow dependencies
└── README.md
```

## Interview preparation

### 30-second project explanation

> “I built a Melbourne weather analysis dashboard using decades of NOAA daily station observations. I trained three models in Python: an LSTM for next-day rainfall, Linear Regression for maximum temperature, and K-Means to group weather conditions. I exported their weights and preprocessing parameters to JSON, then implemented inference in TypeScript so the React and Tailwind app runs as a static site on Netlify without a Python API.”

### Two-minute walkthrough

1. **Problem:** make a long station record explorable and connect it to a small set of understandable machine learning outputs.
2. **Data:** parse NOAA's fixed-width GHCN-Daily format, convert units, make a daily time index, handle missing observations, and encode annual seasonality as cyclic values.
3. **Models:** choose one algorithm for each task—sequence learning for rainfall, an interpretable regression baseline for temperature, and unsupervised clustering for recurring conditions.
4. **Evaluation:** keep the time order intact and use a chronological split to avoid training on the future. Report standard regression metrics for the Linear Regression holdout. Treat clusters as descriptive and the LSTM as an uncalibrated estimate.
5. **Product/deployment:** serialize the fitted parameters and scalers, run inference in typed browser code, and deploy the Vite output as static assets on Netlify.
6. **Tradeoffs:** static inference avoids a server and lowers deployment complexity, but updating a model requires retraining, exporting JSON, committing the assets and redeploying. This demo is not a substitute for meteorological forecasts.

### Common interview questions

**Why an LSTM for rainfall?**

Rainfall prediction is framed as a sequence-to-one problem: use a fixed window of the previous 30 days to estimate the following day's amount. LSTMs can learn ordered patterns across that window. It is a deliberately small baseline; it does not include atmospheric pressure, wind, radar, or probabilistic calibration.

**Why keep Linear Regression if a more complex model might fit better?**

It provides an interpretable comparison point with two explicit inputs. It is fast, easy to explain, and helps make model limitations visible. A useful next experiment would compare it with regularized regression and tree-based models using the same chronological validation strategy.

**How did you avoid time-series leakage?**

Rows remain date ordered. The first 80% of the dates are used to fit the regression and scalers, and the final 20% is reserved from model fitting. Sequence windows only use observations that occur before their target date. Random shuffling is not used for the LSTM training sequence.

**What do MAE, RMSE and R² say here?**

MAE is the average absolute error in °C; RMSE penalizes larger errors more strongly; R² compares explained variance with a constant-mean baseline. The reported MAE of 3.67 °C is a meaningful limitation: the current two-feature regression is a teaching baseline, not a high-accuracy operational forecast.

**Why standardize before K-Means?**

K-Means uses Euclidean distance. Without scaling, the feature's numeric range and units can dominate the distance. Standardizing temperature and precipitation makes their relative variation comparable for clustering.

**Why do the browser models still use Python-trained artifacts?**

Python and TensorFlow/scikit-learn provide the training and preprocessing workflow. The deployed app only needs the fitted weights, coefficients, centroids, and scalers. Exporting those values makes it possible to reproduce the same inference in TypeScript without shipping Python or TensorFlow to every visitor.

**What would you improve next?**

- Add an untouched final evaluation report for the LSTM, including MAE/RMSE and comparison with a seasonal or persistence baseline.
- Evaluate rainfall occurrence separately from rainfall amount and report calibration/uncertainty.
- Add richer predictors such as wind, humidity, pressure and nearby stations, with careful timestamp alignment.
- Compare Linear Regression with nonlinear baselines using walk-forward validation.
- Add automated checks that compare TypeScript inference with the original Python models before exporting a release.
- Add provenance/version metadata and regenerate the published model artifacts on a scheduled pipeline.

## Known limitations

- NOAA's latest record can lag the current date; the dashboard displays the actual newest observation date.
- Precipitation is filled with zero when the source observation is missing. This simplifies the demo but can mislabel unknown rainfall as dry.
- Temperatures are interpolated across short gaps and edge-filled. This is practical for a teaching dashboard, not a quality-controlled climate data product.
- The LSTM target is rainfall amount and can predict positive rainfall on an actually dry day. No forecast uncertainty or probability of rain is shown.
- K-Means assigns every day to a cluster, including unusual extreme conditions; clusters are not authoritative categories.
- The current regression holdout error is several degrees. Treat it as a modeling exercise, not advice for planning or safety.
- Model binaries in `models/` are local and gitignored. Static inference artifacts in `public/` are included for the web build.
