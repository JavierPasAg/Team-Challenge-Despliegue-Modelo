import os
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

os.chdir(os.path.dirname(__file__))

df = pd.read_csv("data/df_alquileres_original.csv", index_col="id")

# ── Bloque 2: Drop de columnas inútiles ──

cols_drop = [
    "listing_url", "scrape_id", "picture_url",
    "host_url", "host_thumbnail_url", "host_picture_url",
    "name", "description", "neighborhood_overview", "host_about", "amenities",
    "host_id", "host_name",
    "calendar_updated", "last_scraped", "calendar_last_scraped",
    "license", "host_neighbourhood", "neighbourhood", "host_location",
    "source",
    "availability_30", "availability_60", "availability_90", "availability_eoy",
    "number_of_reviews", "number_of_reviews_ly", "number_of_reviews_l30d",
    "minimum_minimum_nights", "minimum_maximum_nights",
    "maximum_minimum_nights", "maximum_maximum_nights",
    "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
]

df = df.drop(columns=cols_drop, errors="ignore")

# ── Bloque 3: Duplicados y outliers ──

df = df.drop_duplicates()
df = df[df["estimated_revenue_l365d"] < 1_000_000]

# ── Bloque 4: Transformaciones fijas ──

df = df.dropna(subset=["estimated_revenue_l365d"])
df["revenue_log"] = np.log1p(df["estimated_revenue_l365d"])

# Price: string → numérico
df["price"] = df["price"].astype(str).str.replace(r"[\$,\s]", "", regex=True)
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# host_since → días
df["host_since"] = pd.to_datetime(df["host_since"], errors="coerce")
reference_date = pd.Timestamp("2026-02-27")
df["host_since"] = (reference_date - df["host_since"]).dt.days

# host_response_time → ordinal
response_order = {"within an hour": 0, "within a few hours": 1, "within a day": 2, "a few days or more": 3}
df["host_response_time_ord"] = df["host_response_time"].map(response_order)
df["host_response_time_num"] = df["host_response_time_ord"]
df["has_host_responded"] = df["host_response_time"].notna().astype(int)

# Binarias: t/f → 1/0
bin_map = {"t": 1, "f": 0, "Unknown": 0}
for col in ["host_is_superhost", "host_identity_verified", "host_has_profile_pic", "instant_bookable"]:
    df[col] = df[col].fillna("Unknown").map(bin_map)

# host_response_rate y host_acceptance_rate: "%" → numérico
for col in ["host_response_rate", "host_acceptance_rate"]:
    df[col] = df[col].astype(str).str.replace("%", "").str.strip()
    df[col] = pd.to_numeric(df[col], errors="coerce")

# host_verifications → count
df["host_verifications"] = df["host_verifications"].fillna("[]").apply(ast.literal_eval).apply(len)

# last_review / first_review → features derivadas
df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
df["first_review"] = pd.to_datetime(df["first_review"], errors="coerce")
df["has_reviews"] = df["last_review"].notna().astype(int)
df["days_since_last_review"] = (reference_date - df["last_review"]).dt.days.fillna(9999).astype(int)
df["review_lifetime"] = (df["last_review"] - df["first_review"]).dt.days.fillna(0).astype(int)

# reviews_per_month: NaN → 0
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

# review_scores: NaN → 0
review_cols = [c for c in df.columns if "review_scores" in c]
for col in review_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# bathrooms → numérico + shared
df["bathrooms_num"] = df["bathrooms_text"].str.extract(r"(\d+\.?\d*)").astype(float)
df.loc[df["bathrooms_text"].str.contains("half", case=False, na=False) & df["bathrooms_num"].isna(), "bathrooms_num"] = 0.5
df["is_bathroom_shared"] = df["bathrooms_text"].str.contains("shared", case=False, na=False).astype(int)

# One-hot encoding
df = pd.get_dummies(df, columns=["room_type"], drop_first=True)
df = pd.get_dummies(df, columns=["neighbourhood_group_cleansed"], prefix="ng", drop_first=True)

# Drop columnas residuales
df = df.drop(columns=[
    "last_review", "first_review", "host_response_time",
    "bathrooms_text",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
    "has_availability",
], errors="ignore")

# ── Bloque 5: Train/Test Split ──

X = df.drop(columns=["revenue_log", "estimated_revenue_l365d"])
y = df["revenue_log"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Bloque 6: Imputaciones (estadísticos de train), winsorización y target encoding ──

# Imputación con mediana de train
median_cols = ["price", "beds", "bathrooms", "bedrooms", "bathrooms_num",
               "host_since", "host_response_rate", "host_acceptance_rate",
               "host_listings_count", "host_total_listings_count"]

medians = X_train[median_cols].median()
X_train[median_cols] = X_train[median_cols].fillna(medians)
X_test[median_cols] = X_test[median_cols].fillna(medians)

# host_response_time_ord: NaN → -1 (constante, "sin respuesta")
X_train["host_response_time_ord"] = X_train["host_response_time_ord"].fillna(-1)
X_test["host_response_time_ord"] = X_test["host_response_time_ord"].fillna(-1)
X_train["host_response_time_num"] = X_train["host_response_time_num"].fillna(-1)
X_test["host_response_time_num"] = X_test["host_response_time_num"].fillna(-1)

# Winsorización (percentiles de train)
columns_to_winsor = ["bathrooms", "bedrooms", "beds", "price", "reviews_per_month"]
for col in columns_to_winsor:
    upper = X_train[col].quantile(0.99)
    X_train[col] = X_train[col].clip(upper=upper)
    X_test[col] = X_test[col].clip(upper=upper)

# Target encoding: neighbourhood_cleansed
mean_nh = X_train.join(y_train).groupby("neighbourhood_cleansed")["revenue_log"].mean()
X_train["neighbourhood_revenue"] = X_train["neighbourhood_cleansed"].map(mean_nh)
X_test["neighbourhood_revenue"] = X_test["neighbourhood_cleansed"].map(mean_nh).fillna(y_train.mean())

# Target encoding: property_type
mean_pt = X_train.join(y_train).groupby("property_type")["revenue_log"].mean()
X_train["pt_revenue"] = X_train["property_type"].map(mean_pt)
X_test["pt_revenue"] = X_test["property_type"].map(mean_pt).fillna(y_train.mean())

# Drop columnas usadas para encoding
X_train = X_train.drop(columns=["neighbourhood_cleansed", "property_type"])
X_test = X_test.drop(columns=["neighbourhood_cleansed", "property_type"])

# ── Bloque 7: Optimización, entrenamiento y guardado ──

xgb_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBRegressor(
        random_state=42,
        objective="reg:squarederror",
        verbosity=0
    ))
])

xgb_params = {
    "model__n_estimators": [200, 400, 600],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [3, 5, 7, 10],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0],
    "model__gamma": [0, 0.1, 0.3],
    "model__reg_alpha": [0, 0.1, 1],
    "model__reg_lambda": [1, 1.5, 2]
}

xgb_search = RandomizedSearchCV(
    xgb_pipe,
    xgb_params,
    n_iter=30,
    scoring="neg_root_mean_squared_error",
    cv=5,
    random_state=42,
    n_jobs=-1
)

print("Optimizando hiperparámetros (esto puede tardar unos minutos)...")
xgb_search.fit(X_train, y_train)

print("Mejores parámetros:", xgb_search.best_params_)
print("Mejor RMSE CV:", -xgb_search.best_score_)

model = xgb_search.best_estimator_

y_pred = model.predict(X_test)
rmse_log = root_mean_squared_error(y_test, y_pred)
rmse_eur = root_mean_squared_error(np.expm1(y_test), np.expm1(y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE (log): {rmse_log:.3f}")
print(f"RMSE (€):   {rmse_eur:.2f}")
print(f"R2:         {r2:.4f}")

print("Columnas del modelo:", list(X_train.columns))

joblib.dump(model, "models/modelo_optimizado.pkl")
print("Modelo guardado en models/modelo_optimizado.pkl")