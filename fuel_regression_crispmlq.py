# =============================================================================
#  AIRLINE FUEL-BURN PREDICTION  --  Linear Regression, the CRISP-ML(Q) way
#  Companion code for: Supervised_Learning_Algorithms_and_Metrics.html
# -----------------------------------------------------------------------------
#  HOW TO RUN (VS Code):
#    1. Open this folder in VS Code. Install the "Python" + "Jupyter" extensions.
#    2. Each block below starts with  # %%  --> VS Code shows a "Run Cell" button.
#       Run them ONE BY ONE, top to bottom, and read the printed output.
#    3. See README_SETUP.md for the MySQL + virtual-environment setup.
#
#  CRISP-ML(Q) phases covered:
#    1 Business & Data Understanding | 2 Data Preparation | 3 Model Building (+HPO)
#    4 Evaluation | 5 Deployment | 6 Monitoring & Maintenance
#    (Q) = quality checks are noted inside each phase.
# =============================================================================


# %% ---------------------------------------------------------------------------
# PHASE 0 : Imports & configuration  (run once)
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error, r2_score)
import statsmodels.api as sm
import joblib

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

# ---- MySQL connection details (edit these to match YOUR MySQL) --------------
MYSQL_USER = "root"
MYSQL_PASSWORD = "harsha1174"   # <-- change me
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DB = "airline"                    # create this database once (see README)
TABLE_NAME = "flight_fuel_logs"

CSV_PATH = "flight_fuel_logs.csv"       # the raw file we start from
TARGET = "fuel_kg"                      # what we predict
CATEGORICAL = ["aircraft_type"]
NUMERIC = ["distance_km", "payload_tonnes", "headwind_kts", "cruise_altitude_ft"]


# %% ---------------------------------------------------------------------------
# PHASE 1 : BUSINESS & DATA UNDERSTANDING
#   Goal: predict fuel burn (kg) per flight so dispatchers load the right fuel.
#   Step 1a - load the raw CSV into pandas and look at it.
# -----------------------------------------------------------------------------
raw = pd.read_csv(CSV_PATH)

print("Shape (rows, cols):", raw.shape)     # -> (120, 6)
print("\nFirst 5 rows:")
print(raw.head())
print("\nColumn types & non-nulls:")
print(raw.info())
print("\nSummary statistics:")
print(raw.describe(include="all"))
# WHAT YOU SEE: distance_km ranges ~350-4000, fuel_kg ~2000-14000, one text
# column (aircraft_type = A320/B737). No missing values -> good.


# %% ---------------------------------------------------------------------------
# PHASE 1 : Quick visual understanding -- is the relationship linear?
# -----------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(raw["distance_km"], raw["fuel_kg"], s=18, color="#0080a8")
plt.xlabel("Flight distance (km)")
plt.ylabel("Fuel burned (kg)")
plt.title("Fuel vs Distance -- looks like a straight line")
plt.tight_layout()
plt.show()

print("Correlation of each numeric column with fuel_kg:")
print(raw[NUMERIC + [TARGET]].corr()[TARGET].round(3))
# WHAT YOU SEE: distance_km correlates ~0.99 with fuel -> the dominant driver.


# %% ---------------------------------------------------------------------------
# PHASE 1 : Push the raw data INTO MySQL (one-time load), then work FROM MySQL.
#   In real projects data lives in a database, not a loose CSV. We mirror that:
#   load the CSV into a MySQL table once, then always read from MySQL.
#   (Q) quality: we print the row count MySQL returns to confirm the load.
#   NOTE: quote_plus() safely encodes passwords that contain special characters
#   such as @ # : / -- without it the connection URL would break.
# -----------------------------------------------------------------------------
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

engine = create_engine(
    f"mysql+pymysql://{quote_plus(MYSQL_USER)}:{quote_plus(MYSQL_PASSWORD)}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)

# Write the DataFrame to MySQL (creates/replaces the table). Run this ONCE.
# If MySQL is not running / not configured yet, we fall back to the CSV so the
# rest of the notebook still runs -- set up MySQL later and re-run this cell.
try:
    raw.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
    with engine.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
    MYSQL_OK = True
    print(f"MySQL table '{TABLE_NAME}' now has {n} rows.")   # -> 120
    # TIP: in MySQL Workbench / CLI:  SELECT * FROM airline.flight_fuel_logs;
except Exception as e:
    MYSQL_OK = False
    print("Could NOT reach MySQL -- will use the CSV instead so you can continue.")
    print("  Reason:", str(e).splitlines()[0])
    print("  Fix: (1) start the MySQL server, (2) run  CREATE DATABASE airline;")
    print("       (3) set MYSQL_PASSWORD at the top to your real password.")


# %% ---------------------------------------------------------------------------
# PHASE 1 : Pull the data back FROM MySQL into pandas -- this df drives everything.
#   (Uses MySQL if the previous cell connected; otherwise reads the CSV.)
# -----------------------------------------------------------------------------
if MYSQL_OK:
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)
    print("Pulled from MySQL:", df.shape)
else:
    df = pd.read_csv(CSV_PATH)
    print("Loaded from CSV (MySQL not available):", df.shape)
print(df.head())


# %% ---------------------------------------------------------------------------
# PHASE 2 : DATA PREPARATION
#   Check quality, then define X (inputs) and y (target).
#   (Q) quality: confirm no missing values and correct dtypes before modelling.
# -----------------------------------------------------------------------------
print("Missing values per column:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())

X = df[NUMERIC + CATEGORICAL].copy()   # 5 input features
y = df[TARGET].copy()                  # fuel_kg
print("\nX columns:", list(X.columns))
print("y:", y.name)


# %% ---------------------------------------------------------------------------
# PHASE 2 : Build a preprocessing recipe.
#   - Scale numeric columns (needed so regularization treats them fairly).
#   - One-hot encode the text column aircraft_type (A320/B737 -> 0/1).
#   Wrapping this in a ColumnTransformer keeps preprocessing + model in ONE object
#   so deployment is a single artifact.
# -----------------------------------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ]
)
print(preprocess)


# %% ---------------------------------------------------------------------------
# PHASE 3 : MODEL BUILDING  --  train / test split first (never test on train data)
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("Train:", X_train.shape, " Test:", X_test.shape)   # 96 / 24


# %% ---------------------------------------------------------------------------
# PHASE 3 : The MODEL ESTIMATE OUTPUT (statsmodels OLS).
#   This is the classic regression table -- we use UNSCALED features here so the
#   coefficients read in natural units (kg per km, kg per tonne, ...).
# -----------------------------------------------------------------------------
X_sm = pd.get_dummies(X_train, columns=CATEGORICAL, drop_first=True).astype(float)
X_sm = sm.add_constant(X_sm)                 # adds the intercept term b0
ols = sm.OLS(y_train.astype(float), X_sm).fit()
print(ols.summary())

# ---- HOW TO READ THE OUTPUT (what the values mean) --------------------------
#   coef       : the b's.  distance_km ~= 3.01  -> +1 km burns ~3.01 kg more fuel.
#                payload_tonnes ~= 12.3 -> +1 tonne payload -> +12.3 kg fuel.
#                const ~= 2100 -> fixed baseline (taxi/take-off/reserve) at x=0.
#   std err    : uncertainty on each coef. Smaller = more precise.
#   t          : coef / std err.  Large |t| -> the effect is clearly non-zero.
#   P>|t|      : the p-value. < 0.05 -> feature is statistically significant.
#                (All our features should show 0.000 here -> all matter.)
#   [0.025 0.975] : 95% confidence interval for the coef. If it excludes 0, keep it.
#   R-squared  : share of fuel variation explained. ~0.999 here (distance dominates).
#   Adj. R-sq. : R-squared adjusted for the number of features (fairer comparison).


# %% ---------------------------------------------------------------------------
# PHASE 3 : The same model in scikit-learn (inside the pipeline) -- this is the
#   object we will tune and deploy. Coefficients are on SCALED inputs here.
# -----------------------------------------------------------------------------
linreg = Pipeline([("prep", preprocess), ("model", LinearRegression())])
linreg.fit(X_train, y_train)

feat_names = linreg.named_steps["prep"].get_feature_names_out()
coefs = linreg.named_steps["model"].coef_
print("Intercept:", round(linreg.named_steps["model"].intercept_, 1))
print("Coefficients (on scaled inputs):")
for name, c in zip(feat_names, coefs):
    print(f"   {name:<28} {c:10.2f}")


# %% ---------------------------------------------------------------------------
# PHASE 3 : Regularized versions (avoid over/under-fitting). alpha = penalty dial.
# -----------------------------------------------------------------------------
for name, model in [("Ridge (L2)", Ridge(alpha=1.0)),
                    ("Lasso (L1)", Lasso(alpha=1.0)),
                    ("ElasticNet", ElasticNet(alpha=1.0, l1_ratio=0.5))]:
    pipe = Pipeline([("prep", preprocess), ("model", model)]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(f"{name:<12}  test RMSE = {np.sqrt(mean_squared_error(y_test, pred)):8.1f}")


# %% ---------------------------------------------------------------------------
# PHASE 3 : HYPERPARAMETER OPTIMIZATION (HPO) -- Grid Search over alpha with CV.
#   Every candidate alpha is scored by 5-fold cross-validation on the TRAIN set,
#   so we pick a value that generalises (not one that got lucky on one split).
# -----------------------------------------------------------------------------
grid_pipe = Pipeline([("prep", preprocess), ("model", Ridge())])
param_grid = {"model__alpha": [0.01, 0.1, 1, 10, 100, 300]}

grid = GridSearchCV(grid_pipe, param_grid, cv=5,
                    scoring="neg_root_mean_squared_error")
grid.fit(X_train, y_train)
print("Best alpha:", grid.best_params_)
print("Best CV RMSE:", round(-grid.best_score_, 1))
best_model = grid.best_estimator_          # tuned pipeline, ready to evaluate


# %% ---------------------------------------------------------------------------
# PHASE 3 : (Optional) Smarter HPO with Optuna -- learns from past trials.
#   Run once:  pip install optuna
# -----------------------------------------------------------------------------
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-3, 500, log=True)
        l1 = trial.suggest_float("l1_ratio", 0.0, 1.0)
        pipe = Pipeline([("prep", preprocess),
                         ("model", ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000))])
        score = cross_val_score(pipe, X_train, y_train, cv=5,
                                scoring="neg_root_mean_squared_error").mean()
        return -score            # Optuna minimises -> return RMSE

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=40)
    print("Optuna best params:", study.best_params)
    print("Optuna best RMSE:", round(study.best_value, 1))
except ImportError:
    print("optuna not installed -- run: pip install optuna  (this cell is optional)")


# %% ---------------------------------------------------------------------------
# PHASE 4 : EVALUATION -- score the tuned model on the untouched TEST set.
#   We report several metrics because each answers a different question.
# -----------------------------------------------------------------------------
def report(model, X_te, y_te, label=""):
    p = model.predict(X_te)
    print(f"--- {label} ---")
    print(f"MAE   = {mean_absolute_error(y_te, p):8.1f} kg   (typical miss)")
    print(f"MSE   = {mean_squared_error(y_te, p):10.1f}      (punishes big misses)")
    print(f"RMSE  = {np.sqrt(mean_squared_error(y_te, p)):8.1f} kg   (big-miss, in kg)")
    print(f"MAPE  = {mean_absolute_percentage_error(y_te, p)*100:6.2f} %    (scale-free)")
    print(f"R2    = {r2_score(y_te, p):8.3f}        (variation explained)")
    return p

pred = report(best_model, X_test, y_test, "Tuned Ridge on TEST")
# WHAT YOU SEE (approx): MAE ~90 kg, RMSE ~100 kg, MAPE ~1.5%, R2 ~0.999.


# %% ---------------------------------------------------------------------------
# PHASE 4 : Residual check -- errors should look like random noise around 0.
# -----------------------------------------------------------------------------
resid = y_test - pred
plt.figure(figsize=(6, 4))
plt.scatter(pred, resid, s=18, color="#f38020")
plt.axhline(0, color="#0e1d42", lw=1)
plt.xlabel("Predicted fuel (kg)")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residuals -- no pattern = assumptions look OK")
plt.tight_layout()
plt.show()


# %% ---------------------------------------------------------------------------
# PHASE 5 : DEPLOYMENT -- refit on ALL data and save ONE artifact for production.
#   The saved pipeline contains preprocessing + model, so production only needs
#   to pass raw columns in and gets a prediction out.
# -----------------------------------------------------------------------------
final_model = grid.best_estimator_        # the tuned pipeline
final_model.fit(X, y)                      # train on 100% of the data now
joblib.dump(final_model, "fuel_model.joblib")
print("Saved -> fuel_model.joblib")

# Quick sanity prediction for one new flight (raw inputs, no manual preprocessing):
new_flight = pd.DataFrame([{
    "distance_km": 2200, "payload_tonnes": 14.0, "headwind_kts": 12,
    "cruise_altitude_ft": 35000, "aircraft_type": "A320",
}])
print("Predicted fuel for the new flight:",
      round(float(final_model.predict(new_flight)[0]), 0), "kg")
# -> the model.joblib file is what app.py loads to serve live predictions.


# %% ---------------------------------------------------------------------------
# PHASE 6 : MONITORING & MAINTENANCE (production hygiene)
#   Models drift as the world changes (new aircraft, fuel-mix, routes). Watch it.
# -----------------------------------------------------------------------------
def check_drift(reference: pd.DataFrame, live: pd.DataFrame, col="distance_km"):
    """Very simple drift alarm: has the live mean moved a lot vs training data?"""
    ref_mean, live_mean = reference[col].mean(), live[col].mean()
    shift = abs(live_mean - ref_mean) / ref_mean * 100
    print(f"{col}: train mean={ref_mean:.0f}, live mean={live_mean:.0f}, shift={shift:.1f}%")
    if shift > 20:
        print("  ALERT: input distribution has shifted -> consider retraining.")
    return shift

# Example: compare training data against the latest month pulled from MySQL.
check_drift(df, df.sample(30, random_state=1))
# MAINTENANCE CHECKLIST: schedule monthly re-pull from MySQL, re-evaluate metrics,
# retrain if RMSE/MAPE degrade or drift fires, version each model (date + metrics).
