"""Train AgriConnect+ ML models and write reproducible metrics.

Usage:
    python train_models.py

Outputs:
    models/price_model.joblib
    models/yield_model.joblib
    models/decision_model.joblib
    models/metrics.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    XGBRegressor = None
    HAS_XGBOOST = False

import joblib


MODEL_DIR = Path("models")
RANDOM_STATE = 42

CROP_CYCLES = {
    "Amaranthus": 3, "Banana": 1, "Beans": 2, "Beetroot": 2,
    "Bitter Gourd": 2, "Bottle Gourd": 2, "Brinjal": 2,
    "Cabbage": 2, "Capsicum": 2, "Carrot": 2, "Cauliflower": 2,
    "Coconut": 1, "Cotton": 1, "Garlic": 1, "Grapes": 1,
    "Green Chilli": 2, "Groundnut": 2, "Guava": 1, "Lemon": 1,
    "Maize": 2, "Mango": 1, "Mustard": 1, "Onion": 2,
    "Orange": 1, "Papaya": 1, "Pomegranate": 1, "Potato": 2,
    "Pumpkin": 2, "Rice": 2, "Soyabean": 1, "Spinach": 3,
    "Sugarcane": 1, "Sunflower": 2, "Tomato": 3,
    "Turmeric": 1, "Wheat": 1,
}

FARMING_COSTS = {
    "Amaranthus": 4000, "Banana": 35000, "Beans": 6000,
    "Beetroot": 5000, "Bitter Gourd": 6000, "Bottle Gourd": 5000,
    "Brinjal": 6000, "Cabbage": 6000, "Capsicum": 9000,
    "Carrot": 7500, "Cauliflower": 7000, "Coconut": 20000,
    "Cotton": 18000, "Garlic": 25000, "Grapes": 45000,
    "Green Chilli": 7500, "Groundnut": 9000, "Guava": 20000,
    "Lemon": 18000, "Maize": 6000, "Mango": 20000,
    "Mustard": 10000, "Onion": 9000, "Orange": 20000,
    "Papaya": 22000, "Pomegranate": 35000, "Potato": 10000,
    "Pumpkin": 5000, "Rice": 11000, "Soyabean": 10000,
    "Spinach": 2700, "Sugarcane": 25000, "Sunflower": 5000,
    "Tomato": 6000, "Turmeric": 22000, "Wheat": 12000,
}


def dense_one_hot(columns):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False), columns
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False), columns


def make_regressor(max_depth=8, estimators=150):
    if HAS_XGBOOST:
        return XGBRegressor(
            n_estimators=estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=2,
        )
    return HistGradientBoostingRegressor(max_iter=estimators, max_leaf_nodes=31, random_state=RANDOM_STATE)


def make_classifier():
    if HAS_XGBOOST:
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=2,
        )
    return HistGradientBoostingClassifier(max_iter=200, random_state=RANDOM_STATE)


def train_price_model(price_df):
    features = ["state", "district", "crop"]
    target = "avg_price_quintal"
    train, test = train_test_split(price_df[features + [target]].dropna(), test_size=0.2, random_state=RANDOM_STATE)
    encoder, cols = dense_one_hot(features)
    model = Pipeline([
        ("prep", ColumnTransformer([("cat", encoder, cols)], remainder="drop")),
        ("model", make_regressor(max_depth=10, estimators=150)),
    ])
    model.fit(train[features], train[target])
    pred = model.predict(test[features])
    return model, {
        "algorithm": "XGBRegressor" if HAS_XGBOOST else "HistGradientBoostingRegressor",
        "rows": int(len(price_df)),
        "features": features,
        "r2": float(r2_score(test[target], pred)),
        "mae": float(mean_absolute_error(test[target], pred)),
    }


def train_yield_model(yield_df):
    base = yield_df[["state", "crop", "avg_yield_kg_ha"]].dropna().copy()
    irrigated = base.copy()
    irrigated["irrigation"] = "Irrigated"
    irrigated["target_yield"] = irrigated["avg_yield_kg_ha"] * 1.2
    rainfed = base.copy()
    rainfed["irrigation"] = "Rainfed"
    rainfed["target_yield"] = rainfed["avg_yield_kg_ha"] * 0.8
    data = pd.concat([irrigated, rainfed], ignore_index=True)
    features = ["state", "crop", "irrigation"]
    train, test = train_test_split(data[features + ["target_yield"]], test_size=0.2, random_state=RANDOM_STATE)
    encoder, cols = dense_one_hot(features)
    model = Pipeline([
        ("prep", ColumnTransformer([("cat", encoder, cols)], remainder="drop")),
        ("model", make_regressor(max_depth=8, estimators=150)),
    ])
    model.fit(train[features], train["target_yield"])
    pred = model.predict(test[features])
    return model, {
        "algorithm": "XGBRegressor" if HAS_XGBOOST else "HistGradientBoostingRegressor",
        "base_rows": int(len(yield_df)),
        "augmented_rows": int(len(data)),
        "features": features,
        "r2": float(r2_score(test["target_yield"], pred)),
        "mae": float(mean_absolute_error(test["target_yield"], pred)),
    }


def state_lease_average(state, state_lease):
    rates = state_lease.get(state, {})
    return float(rates.get("avg", 25000))


def build_decision_dataset(price_df, yield_df, state_lease, sample_size=270000):
    rng = np.random.default_rng(RANDOM_STATE)
    usable_prices = price_df[price_df["crop"].isin(CROP_CYCLES)].dropna().copy()
    sampled = usable_prices.sample(n=min(len(usable_prices), 30000), replace=len(usable_prices) < 30000, random_state=RANDOM_STATE)
    yield_lookup = yield_df.groupby(["state", "crop"])["avg_yield_kg_ha"].mean().to_dict()
    crop_yield_lookup = yield_df.groupby("crop")["avg_yield_kg_ha"].mean().to_dict()

    scenarios = [("Low", 0.8), ("Average", 1.0), ("High", 1.2)]
    acres_options = [1, 2, 5]
    years_options = [1, 3, 5]
    irrigation_options = [("Rainfed", 0.8), ("Irrigated", 1.2)]
    rows = []

    for _, r in sampled.iterrows():
        crop = r["crop"]
        base_yield = yield_lookup.get((r["state"], crop), crop_yield_lookup.get(crop, 4000.0))
        for scenario, price_mult in scenarios:
            for acres in acres_options:
                for years in years_options:
                    for irrigation, yield_mult in irrigation_options:
                        price = float(r["avg_price_quintal"]) * price_mult
                        yield_kg = float(base_yield) * yield_mult
                        cycles = CROP_CYCLES.get(crop, 1)
                        cost = FARMING_COSTS.get(crop, 15000)
                        lease = state_lease_average(r["state"], state_lease)
                        yqa = (yield_kg / 100.0) / 2.47
                        total_crop = ((yqa * price - cost) * cycles) * acres * years
                        total_lease = lease * acres * years
                        rows.append({
                            "state": r["state"],
                            "district": r["district"],
                            "crop": crop,
                            "irrigation": irrigation,
                            "market_scenario": scenario,
                            "price_per_quintal": price,
                            "yield_kg_ha": yield_kg,
                            "cost_per_acre_cycle": cost,
                            "cycles_per_year": cycles,
                            "acres": acres,
                            "years": years,
                            "lease_per_acre_year": lease,
                            "label": int(total_crop > total_lease),
                        })
                        if len(rows) >= sample_size:
                            return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def train_decision_model(decision_df):
    features = [
        "state", "district", "crop", "irrigation", "market_scenario",
        "price_per_quintal", "yield_kg_ha", "cost_per_acre_cycle",
        "cycles_per_year", "acres", "years", "lease_per_acre_year",
    ]
    cat = ["state", "district", "crop", "irrigation", "market_scenario"]
    num = [c for c in features if c not in cat]
    train, test = train_test_split(decision_df[features + ["label"]], test_size=0.2, random_state=RANDOM_STATE, stratify=decision_df["label"])
    encoder, cols = dense_one_hot(cat)
    model = Pipeline([
        ("prep", ColumnTransformer([("cat", encoder, cols), ("num", "passthrough", num)])),
        ("model", make_classifier()),
    ])
    model.fit(train[features], train["label"])
    pred = model.predict(test[features])
    cm = confusion_matrix(test["label"], pred).tolist()
    return model, {
        "algorithm": "XGBClassifier" if HAS_XGBOOST else "HistGradientBoostingClassifier",
        "rows": int(len(decision_df)),
        "features": features,
        "accuracy": float(accuracy_score(test["label"], pred)),
        "f1": float(f1_score(test["label"], pred)),
        "confusion_matrix": cm,
        "important_note": (
            "Labels are generated by the deterministic economic engine. "
            "Report this as a decision-consistency model, not real-world farmer preference prediction."
        ),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    price_df = pd.read_csv("crop_prices.csv")
    yield_df = pd.read_csv("crop_yield.csv")
    with open("lease_rates.json", encoding="utf-8") as f:
        state_lease = json.load(f)

    price_model, price_metrics = train_price_model(price_df)
    yield_model, yield_metrics = train_yield_model(yield_df)
    decision_df = build_decision_dataset(price_df, yield_df, state_lease)
    decision_model, decision_metrics = train_decision_model(decision_df)

    joblib.dump(price_model, MODEL_DIR / "price_model.joblib")
    joblib.dump(yield_model, MODEL_DIR / "yield_model.joblib")
    joblib.dump(decision_model, MODEL_DIR / "decision_model.joblib")

    metrics = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "random_state": RANDOM_STATE,
        "xgboost_used": HAS_XGBOOST,
        "price_model": price_metrics,
        "yield_model": yield_metrics,
        "decision_model": decision_metrics,
        "dataset_coverage": {
            "price_rows": int(len(price_df)),
            "yield_rows": int(len(yield_df)),
            "states_in_price_data": int(price_df["state"].nunique()),
            "districts_in_price_data": int(price_df["district"].nunique()),
            "crops_in_price_data": int(price_df["crop"].nunique()),
            "paper_supported_crops_with_costs": int(len(CROP_CYCLES)),
        },
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
