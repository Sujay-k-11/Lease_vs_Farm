"""Model training/inference helpers for AgriConnect+.

The dashboard can run before ML artifacts are trained. In that case these
helpers fall back to transparent data lookup and deterministic formulas, while
showing the source of each value for paper/debugging purposes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:  # pragma: no cover - handled at runtime in Streamlit
    joblib = None


MODEL_DIR = Path("models")
METRICS_PATH = MODEL_DIR / "metrics.json"


def load_artifacts(model_dir: Union[Path, str] = MODEL_DIR) -> Dict[str, Any]:
    """Load model artifacts if present.

    Returns a dictionary with a stable shape so callers can always inspect
    availability and metadata.
    """
    model_dir = Path(model_dir)
    artifacts: Dict[str, Any] = {
        "available": False,
        "price_model": None,
        "yield_model": None,
        "decision_model": None,
        "metrics": {},
        "error": None,
    }

    if joblib is None:
        artifacts["error"] = "joblib is not installed"
        return artifacts

    try:
        price_path = model_dir / "price_model.joblib"
        yield_path = model_dir / "yield_model.joblib"
        decision_path = model_dir / "decision_model.joblib"
        if price_path.exists():
            artifacts["price_model"] = joblib.load(price_path)
        if yield_path.exists():
            artifacts["yield_model"] = joblib.load(yield_path)
        if decision_path.exists():
            artifacts["decision_model"] = joblib.load(decision_path)
        if (model_dir / "metrics.json").exists():
            artifacts["metrics"] = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        artifacts["available"] = any(
            artifacts[k] is not None for k in ("price_model", "yield_model", "decision_model")
        )
    except Exception as exc:  # pragma: no cover - surfaced in UI
        artifacts["error"] = str(exc)

    return artifacts


def lookup_price(price_df: pd.DataFrame, state: str, district: str, crop: str) -> Optional[float]:
    row = price_df[
        (price_df["state"].str.lower() == state.lower())
        & (price_df["district"].str.lower() == district.lower())
        & (price_df["crop"].str.lower() == crop.lower())
    ]
    return float(row["avg_price_quintal"].values[0]) if not row.empty else None


def lookup_yield(yield_df: pd.DataFrame, state: str, crop: str) -> float:
    exact = yield_df[
        (yield_df["state"].str.lower() == state.lower())
        & (yield_df["crop"].str.lower() == crop.lower())
    ]
    if not exact.empty:
        return float(exact["avg_yield_kg_ha"].values[0])

    crop_rows = yield_df[yield_df["crop"].str.lower() == crop.lower()]
    if not crop_rows.empty:
        return float(crop_rows["avg_yield_kg_ha"].mean())

    return 4000.0


def predict_price(
    artifacts: Dict[str, Any],
    price_df: pd.DataFrame,
    state: str,
    district: str,
    crop: str,
    scenario: str,
) -> Dict[str, Any]:
    base_price = None
    source = "fallback_default"

    model = artifacts.get("price_model")
    if model is not None:
        row = pd.DataFrame([{"state": state, "district": district, "crop": crop}])
        try:
            base_price = float(model.predict(row)[0])
            source = "ml_price_model"
        except Exception as exc:
            source = f"lookup_after_model_error: {exc}"

    if base_price is None:
        base_price = lookup_price(price_df, state, district, crop)
        source = "dataset_lookup" if base_price is not None else source

    if base_price is None:
        base_price = 2500.0

    multiplier = 0.8 if "Low" in scenario else 1.2 if "High" in scenario else 1.0
    return {
        "base_price": base_price,
        "price": base_price * multiplier,
        "source": source,
        "scenario_multiplier": multiplier,
    }


def predict_yield(
    artifacts: Dict[str, Any],
    yield_df: pd.DataFrame,
    state: str,
    crop: str,
    irrigation: str,
) -> Dict[str, Any]:
    adjusted_yield = None
    source = "fallback_default"

    model = artifacts.get("yield_model")
    if model is not None:
        row = pd.DataFrame([{"state": state, "crop": crop, "irrigation": irrigation}])
        try:
            adjusted_yield = float(model.predict(row)[0])
            source = "ml_yield_model"
        except Exception as exc:
            source = f"lookup_after_model_error: {exc}"

    if adjusted_yield is None:
        base = lookup_yield(yield_df, state, crop)
        adjusted_yield = base * (0.8 if irrigation == "Rainfed" else 1.2)
        source = "dataset_lookup_with_irrigation_multiplier"

    return {"yield_kg_ha": adjusted_yield, "source": source}


def compute_income(
    price_per_quintal: float,
    yield_kg_ha: float,
    cost_per_acre_cycle: float,
    cycles_per_year: int,
    acres: float,
    years: float,
    lease_per_acre_year: float,
) -> Dict[str, float]:
    yield_quintal_acre = (yield_kg_ha / 100.0) / 2.47
    gross_cycle = yield_quintal_acre * price_per_quintal
    net_cycle = gross_cycle - cost_per_acre_cycle
    net_year = net_cycle * cycles_per_year
    total_crop = net_year * acres * years
    total_lease = lease_per_acre_year * acres * years
    total_diff = total_crop - total_lease
    margin_pct = abs(total_diff / total_lease) * 100 if total_lease > 0 else 0.0
    return {
        "yield_quintal_acre": yield_quintal_acre,
        "gross_cycle": gross_cycle,
        "net_cycle": net_cycle,
        "net_year": net_year,
        "total_crop": total_crop,
        "total_lease": total_lease,
        "total_diff": total_diff,
        "margin_pct": margin_pct,
    }


def predict_decision(
    artifacts: Dict[str, Any],
    features: Dict[str, Any],
    deterministic_better: str,
) -> Dict[str, Any]:
    model = artifacts.get("decision_model")
    if model is None:
        return {
            "decision": deterministic_better,
            "confidence": None,
            "source": "deterministic_economic_engine",
        }

    row = pd.DataFrame([features])
    try:
        pred = int(model.predict(row)[0])
        decision = "Crop Farming" if pred == 1 else "Leasing"
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(row)[0]))
        return {"decision": decision, "confidence": confidence, "source": "ml_decision_classifier"}
    except Exception as exc:
        return {
            "decision": deterministic_better,
            "confidence": None,
            "source": f"deterministic_after_model_error: {exc}",
        }
