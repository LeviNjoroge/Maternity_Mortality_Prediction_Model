"""
Notebook-Driven Maternal Mortality Backend Service

This backend uses the notebook `Maternity_Mortality_Prediction_Model.ipynb`
as the source for key configuration values (DATA_PATH, country_name, forecast_steps)
and recreates the core model workflow in Python for API-based analytics.

Run:
  python notebook_backend_service.py

Optional args:
  --notebook Maternity_Mortality_Prediction_Model.ipynb
  --host 127.0.0.1
  --port 5000
  --debug
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


YEAR_PATTERN = re.compile(r"\((19|20)\d{2}\)$")


@dataclass
class NotebookConfig:
    notebook_path: Path
    data_path: Path
    default_country: str
    forecast_steps: int


class NotebookSourceExtractor:
    def __init__(self, notebook_path: Path) -> None:
        self.notebook_path = notebook_path
        self.notebook_json: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        # utf-8-sig protects against BOM-prefixed notebook files.
        content = self.notebook_path.read_text(encoding="utf-8-sig")
        self.notebook_json = json.loads(content)
        return self.notebook_json

    def _collect_code_text(self) -> str:
        if not self.notebook_json:
            self.load()

        cells = self.notebook_json.get("cells", [])
        chunks: List[str] = []
        for cell in cells:
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", [])
            if isinstance(source, list):
                chunks.append("\n".join(str(line) for line in source))
            else:
                chunks.append(str(source))

        return "\n\n".join(chunks)

    def extract_string_literal(self, variable_name: str) -> Optional[str]:
        text = self._collect_code_text()
        pattern = re.compile(
            rf"\b{re.escape(variable_name)}\s*=\s*['\"]([^'\"]+)['\"]"
        )
        match = pattern.search(text)
        return match.group(1) if match else None

    def extract_integer_literal(self, variable_name: str) -> Optional[int]:
        text = self._collect_code_text()
        pattern = re.compile(rf"\b{re.escape(variable_name)}\s*=\s*(\d+)")
        match = pattern.search(text)
        return int(match.group(1)) if match else None


def score_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_true_arr, y_pred_arr)
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mse)),
        "mse": float(mse),
        "r2": float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 1 else float("nan"),
    }


class MaternalMortalityBackend:
    def __init__(self, config: NotebookConfig) -> None:
        self.config = config

        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.mmr_long: pd.DataFrame = pd.DataFrame()
        self.model_df: pd.DataFrame = pd.DataFrame()

        self.year_cols: List[str] = []
        self.id_cols: List[str] = []
        self.undp_col: Optional[str] = None

        self.continent_encoder = LabelEncoder()
        self.undp_encoder = LabelEncoder()

        self.lr_simple: Optional[LinearRegression] = None
        self.lr_multi: Optional[LinearRegression] = None
        self.lr_scaler: Optional[StandardScaler] = None

        self.rf_model: Optional[RandomForestRegressor] = None
        self.xgb_model: Optional[Any] = None

        self.rf_feature_cols: List[str] = [
            "Year",
            "HDI Rank (2021)",
            "lag_1",
            "lag_2",
            "lag_3",
            "rolling_mean_5",
            "mmr_change",
            "year_norm",
            "continent_enc",
            "hdi_group_enc",
            "undp_enc",
        ]

        self.metrics: Dict[str, Dict[str, float]] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}

    def initialize(self) -> None:
        self._load_data()
        self._prepare_shared_data()
        self._train_linear_regression_models()
        self._train_random_forest()
        self._train_xgboost()
        self._fit_default_arima()

    def _load_data(self) -> None:
        if not self.config.data_path.exists():
            raise FileNotFoundError(
                f"Dataset path not found: {self.config.data_path}. "
                "Ensure DATA_PATH in notebook is correct or place CSV in the project folder."
            )

        self.raw_df = pd.read_csv(self.config.data_path)

        # Normalize typo if present in source CSV.
        if "UNDP Developeing Regions" in self.raw_df.columns and "UNDP Developing Regions" not in self.raw_df.columns:
            self.raw_df = self.raw_df.rename(
                columns={"UNDP Developeing Regions": "UNDP Developing Regions"}
            )

    def _prepare_shared_data(self) -> None:
        self.year_cols = [
            c
            for c in self.raw_df.columns
            if YEAR_PATTERN.search(str(c)) and "Maternal Mortality" in str(c)
        ]
        if not self.year_cols:
            raise ValueError("No year-based MMR columns found in dataset.")

        self.undp_col = next((c for c in self.raw_df.columns if "UNDP" in str(c)), None)
        if self.undp_col is None:
            raise ValueError("Could not find UNDP region column in dataset.")

        self.id_cols = [
            col
            for col in [
                "ISO3",
                "Country",
                "Continent",
                "Hemisphere",
                "Human Development Groups",
                self.undp_col,
                "HDI Rank (2021)",
            ]
            if col in self.raw_df.columns
        ]

        self.mmr_long = self.raw_df.melt(
            id_vars=self.id_cols,
            value_vars=self.year_cols,
            var_name="MMR_Year_Column",
            value_name="MMR",
        )

        self.mmr_long["Year"] = self.mmr_long["MMR_Year_Column"].str.extract(r"(\d{4})").astype(int)
        self.mmr_long["MMR"] = pd.to_numeric(self.mmr_long["MMR"], errors="coerce")

        if "HDI Rank (2021)" in self.mmr_long.columns:
            self.mmr_long["HDI Rank (2021)"] = pd.to_numeric(
                self.mmr_long["HDI Rank (2021)"], errors="coerce"
            )
            self.mmr_long["HDI Rank (2021)"] = self.mmr_long["HDI Rank (2021)"].fillna(
                self.mmr_long["HDI Rank (2021)"].median()
            )

        if "Human Development Groups" in self.mmr_long.columns:
            self.mmr_long["Human Development Groups"] = self.mmr_long[
                "Human Development Groups"
            ].fillna("Unknown")

        if "Continent" in self.mmr_long.columns:
            self.mmr_long["Continent"] = self.mmr_long["Continent"].fillna("Unknown")

        self.mmr_long[self.undp_col] = self.mmr_long[self.undp_col].fillna("Unknown")

        hdi_map = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}
        self.mmr_long["hdi_group_enc"] = self.mmr_long["Human Development Groups"].map(hdi_map).fillna(1).astype(int)

        self.mmr_long["continent_enc"] = self.continent_encoder.fit_transform(
            self.mmr_long["Continent"].astype(str)
        )
        self.mmr_long["undp_enc"] = self.undp_encoder.fit_transform(
            self.mmr_long[self.undp_col].astype(str)
        )

        self.mmr_long = self.mmr_long.sort_values(["ISO3", "Year"]).reset_index(drop=True)

        self.mmr_long["lag_1"] = self.mmr_long.groupby("ISO3")["MMR"].shift(1)
        self.mmr_long["lag_2"] = self.mmr_long.groupby("ISO3")["MMR"].shift(2)
        self.mmr_long["lag_3"] = self.mmr_long.groupby("ISO3")["MMR"].shift(3)

        self.mmr_long["rolling_mean_5"] = self.mmr_long.groupby("ISO3")["MMR"].transform(
            lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
        )
        self.mmr_long["mmr_change"] = self.mmr_long.groupby("ISO3")["MMR"].diff()
        self.mmr_long["year_norm"] = (
            (self.mmr_long["Year"] - self.mmr_long["Year"].min())
            / (self.mmr_long["Year"].max() - self.mmr_long["Year"].min())
        )

        self.model_df = self.mmr_long.dropna(
            subset=["MMR", "lag_1", "lag_2", "lag_3", "rolling_mean_5", "mmr_change"]
        ).copy()

    def _train_linear_regression_models(self) -> None:
        mmr_columns = [c for c in self.raw_df.columns if "Maternal Mortality" in str(c)]
        if len(mmr_columns) < 2:
            raise ValueError("Not enough maternal mortality columns for linear regression.")

        x_cols = mmr_columns[:-1]
        y_col = mmr_columns[-1]

        x_simple = self.raw_df[[x_cols[-1]]].copy()
        y_simple = pd.to_numeric(self.raw_df[y_col], errors="coerce")
        valid_simple = ~y_simple.isna() & ~pd.to_numeric(x_simple.iloc[:, 0], errors="coerce").isna()

        x_simple = x_simple[valid_simple].reset_index(drop=True)
        y_simple = y_simple[valid_simple].reset_index(drop=True)

        x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(
            x_simple, y_simple, test_size=0.2, random_state=42
        )

        self.lr_simple = LinearRegression()
        self.lr_simple.fit(x_train_s, y_train_s)
        y_pred_s = self.lr_simple.predict(x_test_s)
        self.metrics["linear_regression_simple"] = score_regression(y_test_s, y_pred_s)

        x_multi = self.raw_df[x_cols].copy()
        x_multi = x_multi.apply(pd.to_numeric, errors="coerce")
        x_multi = x_multi.fillna(x_multi.median())

        if "HDI Rank (2021)" in self.raw_df.columns:
            hdi = pd.to_numeric(self.raw_df["HDI Rank (2021)"], errors="coerce")
            hdi = hdi.fillna(hdi.mean())
            x_multi = pd.concat([x_multi, hdi.rename("HDI Rank (2021)")], axis=1)

        y_multi = pd.to_numeric(self.raw_df[y_col], errors="coerce")
        valid_multi = ~y_multi.isna()

        x_multi = x_multi[valid_multi].reset_index(drop=True)
        y_multi = y_multi[valid_multi].reset_index(drop=True)

        x_train, x_test, y_train, y_test = train_test_split(
            x_multi, y_multi, test_size=0.2, random_state=42
        )

        self.lr_scaler = StandardScaler()
        x_train_scaled = self.lr_scaler.fit_transform(x_train)
        x_test_scaled = self.lr_scaler.transform(x_test)

        self.lr_multi = LinearRegression()
        self.lr_multi.fit(x_train_scaled, y_train)
        y_pred_multi = self.lr_multi.predict(x_test_scaled)

        self.metrics["linear_regression_multi"] = score_regression(y_test, y_pred_multi)

        coeff_df = pd.DataFrame(
            {
                "feature": x_train.columns,
                "coefficient": self.lr_multi.coef_,
                "abs_coefficient": np.abs(self.lr_multi.coef_),
            }
        ).sort_values("abs_coefficient", ascending=False)

        self.feature_importance["linear_regression"] = {
            str(row.feature): float(row.abs_coefficient)
            for _, row in coeff_df.head(30).iterrows()
        }

    def _train_random_forest(self) -> None:
        train_df = self.model_df[self.model_df["Year"] <= 2017].copy()
        test_df = self.model_df[self.model_df["Year"] > 2017].copy()

        if train_df.empty or test_df.empty:
            train_df, test_df = train_test_split(self.model_df, test_size=0.2, random_state=42)

        x_train = train_df[self.rf_feature_cols]
        y_train = train_df["MMR"]
        x_test = test_df[self.rf_feature_cols]
        y_test = test_df["MMR"]

        self.rf_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.rf_model.fit(x_train, y_train)

        y_pred = self.rf_model.predict(x_test)
        self.metrics["random_forest"] = score_regression(y_test, y_pred)

        rf_importance = pd.DataFrame(
            {
                "feature": self.rf_feature_cols,
                "importance": self.rf_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        self.feature_importance["random_forest"] = {
            str(row.feature): float(row.importance)
            for _, row in rf_importance.head(30).iterrows()
        }

    def _train_xgboost(self) -> None:
        if not XGBOOST_AVAILABLE:
            self.metrics["xgboost"] = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "mse": float("nan"),
                "r2": float("nan"),
            }
            self.feature_importance["xgboost"] = {}
            return

        train_df = self.model_df[self.model_df["Year"] <= 2017].copy()
        test_df = self.model_df[self.model_df["Year"] > 2017].copy()

        if train_df.empty or test_df.empty:
            train_df, test_df = train_test_split(self.model_df, test_size=0.2, random_state=42)

        x_train = train_df[self.rf_feature_cols]
        y_train = train_df["MMR"]
        x_test = test_df[self.rf_feature_cols]
        y_test = test_df["MMR"]

        self.xgb_model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
        )

        self.xgb_model.fit(x_train, y_train)
        y_pred = self.xgb_model.predict(x_test)
        self.metrics["xgboost"] = score_regression(y_test, y_pred)

        xgb_importance = pd.DataFrame(
            {
                "feature": self.rf_feature_cols,
                "importance": self.xgb_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        self.feature_importance["xgboost"] = {
            str(row.feature): float(row.importance)
            for _, row in xgb_importance.head(30).iterrows()
        }

    def _country_series(self, country: str) -> pd.Series:
        row = self.raw_df[self.raw_df["Country"] == country]
        if row.empty:
            return pd.Series(dtype=float)

        values = row[self.year_cols].iloc[0]
        numeric = pd.to_numeric(values, errors="coerce")

        # Use a dict to guarantee unique years even if unexpected duplicate columns exist.
        year_value_map: Dict[int, float] = {}
        for col, val in numeric.items():
            match = re.search(r"(\d{4})", str(col))
            if not match:
                continue

            year = int(match.group(1))
            if pd.isna(val):
                continue

            year_value_map[year] = float(val)

        if not year_value_map:
            return pd.Series(dtype=float)

        series = pd.Series(year_value_map, dtype=float)
        return series.sort_index()

    def _fit_default_arima(self) -> None:
        series = self._country_series(self.config.default_country)
        if len(series) < 8:
            self.metrics["arima"] = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "mse": float("nan"),
                "r2": float("nan"),
                "adf_pvalue": float("nan"),
            }
            return

        adf_stat, adf_pvalue, _, _, _, _ = adfuller(series)

        # Simple backtest with the final forecast_steps years.
        steps = min(self.config.forecast_steps, max(1, len(series) // 4))
        train = series.iloc[:-steps]
        test = series.iloc[-steps:]

        if len(train) < 6:
            self.metrics["arima"] = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "mse": float("nan"),
                "r2": float("nan"),
                "adf_pvalue": float(adf_pvalue),
            }
            return

        model = ARIMA(train, order=(1, 1, 1)).fit()
        pred = model.forecast(steps=len(test))
        scores = score_regression(test, pred.to_numpy())
        scores["adf_pvalue"] = float(adf_pvalue)
        self.metrics["arima"] = scores

    def get_countries(self) -> List[str]:
        return sorted(self.raw_df["Country"].dropna().astype(str).unique().tolist())

    def get_country_history(self, country: str) -> List[Dict[str, float]]:
        series = self._country_series(country)
        return [{"year": int(y), "mmr": float(v)} for y, v in series.items()]

    def _build_feature_row(self, country: str, target_year: int) -> Optional[pd.DataFrame]:
        # Build a feature row approximating notebook feature engineering for target_year prediction.
        series = self._country_series(country)
        if series.empty:
            return None

        available_years = series.index.tolist()
        latest_year = max(available_years)

        # If future year requested, use latest known year as anchor.
        anchor_year = min(target_year - 1, latest_year)

        if anchor_year not in series.index:
            return None

        def safe_get(year: int) -> float:
            if year in series.index:
                return float(series.loc[year])
            return float("nan")

        lag_1 = safe_get(anchor_year)
        lag_2 = safe_get(anchor_year - 1)
        lag_3 = safe_get(anchor_year - 2)
        lag_block = [v for v in [safe_get(anchor_year - i) for i in range(0, 5)] if np.isfinite(v)]

        if not (np.isfinite(lag_1) and np.isfinite(lag_2) and np.isfinite(lag_3)):
            return None

        rolling_mean_5 = float(np.mean(lag_block)) if lag_block else float(lag_1)
        mmr_change = float(lag_1 - lag_2) if np.isfinite(lag_2) else 0.0

        country_row = self.raw_df[self.raw_df["Country"] == country].iloc[0]

        continent_value = str(country_row.get("Continent", "Unknown"))
        hdi_group = str(country_row.get("Human Development Groups", "Unknown"))
        undp_value = str(country_row.get(self.undp_col, "Unknown"))
        hdi_rank = pd.to_numeric(country_row.get("HDI Rank (2021)", np.nan), errors="coerce")

        hdi_map = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}

        # Safe encoding even for unseen values.
        known_continents = set(self.continent_encoder.classes_.tolist())
        known_undp = set(self.undp_encoder.classes_.tolist())

        continent_enc = (
            int(self.continent_encoder.transform([continent_value])[0])
            if continent_value in known_continents
            else 0
        )
        undp_enc = (
            int(self.undp_encoder.transform([undp_value])[0])
            if undp_value in known_undp
            else 0
        )

        row = {
            "Year": int(target_year - 1),
            "HDI Rank (2021)": float(hdi_rank) if np.isfinite(hdi_rank) else float(self.mmr_long["HDI Rank (2021)"].median()),
            "lag_1": float(lag_1),
            "lag_2": float(lag_2),
            "lag_3": float(lag_3),
            "rolling_mean_5": float(rolling_mean_5),
            "mmr_change": float(mmr_change),
            "year_norm": float((target_year - self.model_df["Year"].min()) / (self.model_df["Year"].max() - self.model_df["Year"].min())),
            "continent_enc": float(continent_enc),
            "hdi_group_enc": float(hdi_map.get(hdi_group, 1)),
            "undp_enc": float(undp_enc),
        }

        return pd.DataFrame([row], columns=self.rf_feature_cols)

    def _predict_with_linear_simple_iterative(self, country: str, target_year: int) -> float:
        if self.lr_simple is None:
            return float("nan")

        series = self._country_series(country)
        if series.empty:
            return float("nan")

        def scalar(value: Any) -> float:
            if isinstance(value, pd.Series):
                if value.empty:
                    return float("nan")
                value = value.iloc[-1]
            return float(value)

        latest_year = int(series.index.max())
        current_value = scalar(series.loc[latest_year])

        if target_year <= latest_year:
            if target_year in series.index:
                return scalar(series.loc[target_year])
            return current_value

        coef = float(self.lr_simple.coef_[0])
        intercept = float(self.lr_simple.intercept_)

        for _ in range(target_year - latest_year):
            current_value = intercept + coef * current_value

        return float(max(0.0, current_value))

    def _predict_with_rf(self, country: str, target_year: int) -> float:
        if self.rf_model is None:
            return float("nan")
        feature_row = self._build_feature_row(country, target_year)
        if feature_row is None:
            return float("nan")
        return float(max(0.0, self.rf_model.predict(feature_row)[0]))

    def _predict_with_xgb(self, country: str, target_year: int) -> float:
        if self.xgb_model is None:
            return float("nan")
        feature_row = self._build_feature_row(country, target_year)
        if feature_row is None:
            return float("nan")
        return float(max(0.0, self.xgb_model.predict(feature_row)[0]))

    def _predict_with_arima(self, country: str, target_year: int) -> float:
        series = self._country_series(country)
        if len(series) < 8:
            return float("nan")

        latest_year = int(series.index.max())
        if target_year <= latest_year and target_year in series.index:
            return float(series.loc[target_year])

        steps = max(1, target_year - latest_year)
        model = ARIMA(series, order=(1, 1, 1)).fit()
        pred = model.forecast(steps=steps)
        return float(max(0.0, pred.iloc[-1]))

    def predict(self, country: str, target_year: int) -> Dict[str, Any]:
        series = self._country_series(country)
        if series.empty:
            raise ValueError(f"Country '{country}' not found or has no valid MMR history.")

        algo_preds = {
            "linear_regression": self._predict_with_linear_simple_iterative(country, target_year),
            "random_forest": self._predict_with_rf(country, target_year),
            "xgboost": self._predict_with_xgb(country, target_year),
            "arima": self._predict_with_arima(country, target_year),
        }

        valid_values = [v for v in algo_preds.values() if np.isfinite(v)]
        if not valid_values:
            raise ValueError("No model produced a valid prediction for this request.")

        # Weighted ensemble using inverse RMSE when available.
        weighted_sum = 0.0
        weight_sum = 0.0

        model_rmse_map = {
            "linear_regression": self.metrics.get("linear_regression_simple", {}).get("rmse"),
            "random_forest": self.metrics.get("random_forest", {}).get("rmse"),
            "xgboost": self.metrics.get("xgboost", {}).get("rmse"),
            "arima": self.metrics.get("arima", {}).get("rmse"),
        }

        for model_name, pred in algo_preds.items():
            if not np.isfinite(pred):
                continue
            rmse = model_rmse_map.get(model_name)
            if rmse is None or (not np.isfinite(rmse)) or rmse <= 0:
                weight = 1.0
            else:
                weight = 1.0 / float(rmse)

            weighted_sum += float(pred) * weight
            weight_sum += weight

        ensemble = weighted_sum / weight_sum if weight_sum > 0 else float(np.mean(valid_values))

        lower = float(np.percentile(valid_values, 20))
        upper = float(np.percentile(valid_values, 80))

        # Risk bands based on final historical year distribution.
        final_col = sorted(self.year_cols)[-1]
        dist = pd.to_numeric(self.raw_df[final_col], errors="coerce").dropna()
        q33 = float(dist.quantile(0.33)) if not dist.empty else 70.0
        q66 = float(dist.quantile(0.66)) if not dist.empty else 180.0

        if ensemble > q66:
            risk = "High Risk"
        elif ensemble > q33:
            risk = "Mid Risk"
        else:
            risk = "Low Risk"

        return {
            "country": country,
            "target_year": int(target_year),
            "prediction": {
                "linear_regression": float(algo_preds["linear_regression"])
                if np.isfinite(algo_preds["linear_regression"])
                else None,
                "random_forest": float(algo_preds["random_forest"])
                if np.isfinite(algo_preds["random_forest"])
                else None,
                "xgboost": float(algo_preds["xgboost"])
                if np.isfinite(algo_preds["xgboost"])
                else None,
                "arima": float(algo_preds["arima"]) if np.isfinite(algo_preds["arima"]) else None,
            },
            "ensemble": {
                "value": float(ensemble),
                "lower_bound": float(max(0.0, lower)),
                "upper_bound": float(max(0.0, upper)),
            },
            "risk_band": risk,
            "benchmark_rmse": {
                key: (float(val) if val is not None and np.isfinite(val) else None)
                for key, val in model_rmse_map.items()
            },
        }


def build_config_from_notebook(notebook_path: Path) -> NotebookConfig:
    extractor = NotebookSourceExtractor(notebook_path)
    extractor.load()

    data_literal = extractor.extract_string_literal("DATA_PATH") or "Maternal_Mortality.csv"
    country_literal = extractor.extract_string_literal("country_name") or "Kenya"
    steps_literal = extractor.extract_integer_literal("forecast_steps") or 5

    data_path = Path(data_literal)
    if not data_path.is_absolute():
        data_path = (notebook_path.parent / data_path).resolve()

    return NotebookConfig(
        notebook_path=notebook_path.resolve(),
        data_path=data_path,
        default_country=country_literal,
        forecast_steps=int(steps_literal),
    )


def create_app(service: MaternalMortalityBackend) -> Flask:
    app = Flask(__name__)

    @app.after_request
    def _cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

    @app.route("/api/health", methods=["GET"])
    def health() -> Any:
        return jsonify(
            {
                "status": "ok",
                "backend": "notebook-driven",
                "xgboost_available": XGBOOST_AVAILABLE,
                "countries": len(service.get_countries()),
            }
        )

    @app.route("/api/source", methods=["GET"])
    def source() -> Any:
        return jsonify(
            {
                "notebook_path": str(service.config.notebook_path),
                "dataset_path": str(service.config.data_path),
                "default_country": service.config.default_country,
                "forecast_steps": service.config.forecast_steps,
            }
        )

    @app.route("/api/metrics", methods=["GET"])
    def metrics() -> Any:
        return jsonify(
            {
                "metrics": service.metrics,
                "feature_importance": service.feature_importance,
            }
        )

    @app.route("/api/countries", methods=["GET"])
    def countries() -> Any:
        return jsonify({"countries": service.get_countries()})

    @app.route("/api/country-series", methods=["GET"])
    def country_series() -> Any:
        country = request.args.get("country", service.config.default_country)
        points = service.get_country_history(country)
        if not points:
            return jsonify({"error": f"No data found for country '{country}'."}), 404

        return jsonify({"country": country, "series": points})

    @app.route("/api/predict", methods=["GET"])
    def predict() -> Any:
        country = request.args.get("country", service.config.default_country)
        target_year_raw = request.args.get("year")

        if target_year_raw is None:
            # default is one year ahead of latest available.
            sample_series = service._country_series(country)
            if sample_series.empty:
                return jsonify({"error": f"No series available for country '{country}'."}), 404
            target_year = int(sample_series.index.max()) + 1
        else:
            try:
                target_year = int(target_year_raw)
            except ValueError:
                return jsonify({"error": "Query param 'year' must be an integer."}), 400

        try:
            payload = service.predict(country=country, target_year=target_year)
            return jsonify(payload)
        except ValueError as ex:
            return jsonify({"error": str(ex)}), 400
        except Exception as ex:
            return jsonify({"error": f"Prediction failed: {str(ex)}"}), 500

    @app.route("/api/feature-importance", methods=["GET"])
    def feature_importance() -> Any:
        model = request.args.get("model", "random_forest")
        top_n = int(request.args.get("top_n", 15))

        available = service.feature_importance.get(model)
        if available is None:
            return jsonify(
                {
                    "error": "Unknown model.",
                    "allowed_models": list(service.feature_importance.keys()),
                }
            ), 400

        sorted_items = sorted(available.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return jsonify({"model": model, "feature_importance": [{"feature": k, "value": v} for k, v in sorted_items]})

    @app.route("/api/retrain", methods=["POST"])
    def retrain() -> Any:
        try:
            service.initialize()
            return jsonify({"status": "retrained"})
        except Exception as ex:
            return jsonify({"error": f"Retrain failed: {str(ex)}"}), 500

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Notebook-driven backend service for maternal mortality analytics.")
    parser.add_argument(
        "--notebook",
        type=str,
        default="Maternity_Mortality_Prediction_Model.ipynb",
        help="Path to the source notebook.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Flask host")
    parser.add_argument("--port", type=int, default=5000, help="Flask port")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook_path = Path(args.notebook)
    if not notebook_path.is_absolute():
        notebook_path = (Path(__file__).resolve().parent / notebook_path).resolve()

    config = build_config_from_notebook(notebook_path)
    service = MaternalMortalityBackend(config)
    service.initialize()

    app = create_app(service)

    print("Notebook backend service initialized")
    print(f"Notebook source : {config.notebook_path}")
    print(f"Dataset source  : {config.data_path}")
    print(f"Default country : {config.default_country}")
    print(f"Forecast steps  : {config.forecast_steps}")
    print(f"XGBoost enabled : {XGBOOST_AVAILABLE}")
    print(f"Server URL      : http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
