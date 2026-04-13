"""
Unified Maternal Mortality Prediction Pipeline

This script combines four algorithms in one generalized workflow:
1) Linear Regression
2) Random Forest (regression + risk classification + multi-horizon forecasting)
3) XGBoost (lag-feature temporal forecasting)
4) ARIMA (country-level time-series forecasting)

The pipeline is built for the repository dataset `Maternal_Mortality.csv` and is
intended to run as a standalone reproducible analysis module.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.arima.model import ARIMA


YEAR_REGEX = re.compile(r"\((\d{4})\)\s*$")


@dataclass
class UnifiedConfig:
    dataset_path: Path
    output_dir: Path
    target_year: Optional[int] = None
    test_size: float = 0.2
    random_state: int = 42
    horizons: Tuple[int, ...] = (1, 3, 5, 10)
    max_lag: int = 5
    holdout_years: int = 4
    arima_order: Tuple[int, int, int] = (1, 1, 1)
    arima_forecast_horizon: int = 5
    arima_top_countries: int = 12
    arima_country: Optional[str] = None
    make_plots: bool = True


def _as_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _as_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    return value


def safe_json_dump(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_as_serializable(payload), f, indent=2)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true_arr, y_pred_arr)

    metrics = {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
    }

    if len(y_true_arr) > 1:
        metrics["r2"] = float(r2_score(y_true_arr, y_pred_arr))
    else:
        metrics["r2"] = float("nan")

    return metrics


def parse_year_map(columns: Sequence[str]) -> Dict[int, str]:
    year_map: Dict[int, str] = {}
    for col in columns:
        match = YEAR_REGEX.search(str(col))
        if match:
            year_map[int(match.group(1))] = col

    if not year_map:
        raise ValueError("No year-based Maternal Mortality columns were detected.")

    return dict(sorted(year_map.items(), key=lambda kv: kv[0]))


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def plot_actual_vs_pred(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    title: str,
    output_path: Path,
) -> None:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true_arr, y_pred_arr, alpha=0.75)

    min_value = float(min(np.min(y_true_arr), np.min(y_pred_arr)))
    max_value = float(max(np.max(y_true_arr), np.max(y_pred_arr)))
    plt.plot([min_value, max_value], [min_value, max_value], "r--", linewidth=1.5)

    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def plot_top_features(feature_scores: pd.Series, title: str, output_path: Path, top_n: int = 20) -> None:
    top = feature_scores.head(top_n).sort_values(ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(top.index, top.values)
    plt.title(title)
    plt.xlabel("Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


class UnifiedMaternalMortalityPipeline:
    def __init__(self, config: UnifiedConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.df: pd.DataFrame = pd.DataFrame()
        self.year_map: Dict[int, str] = {}
        self.categorical_meta_cols: List[str] = []
        self.numeric_meta_cols: List[str] = []

        self.results: Dict[str, Any] = {}

    def load_and_profile_dataset(self) -> Dict[str, Any]:
        if not self.config.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{self.config.dataset_path}'. "
                "Use --data to provide the correct path."
            )

        self.df = pd.read_csv(self.config.dataset_path)

        # Normalize known typo seen in this dataset.
        if "UNDP Developeing Regions" in self.df.columns:
            self.df = self.df.rename(columns={"UNDP Developeing Regions": "UNDP Developing Regions"})

        self.year_map = parse_year_map(self.df.columns)

        for year_col in self.year_map.values():
            self.df[year_col] = pd.to_numeric(self.df[year_col], errors="coerce")

        if "HDI Rank (2021)" in self.df.columns:
            self.df["HDI Rank (2021)"] = pd.to_numeric(self.df["HDI Rank (2021)"], errors="coerce")

        self.categorical_meta_cols = [
            col
            for col in [
                "Continent",
                "Hemisphere",
                "Human Development Groups",
                "UNDP Developing Regions",
            ]
            if col in self.df.columns
        ]

        self.numeric_meta_cols = [col for col in ["HDI Rank (2021)"] if col in self.df.columns]

        target_year = self.config.target_year or max(self.year_map)
        if target_year not in self.year_map:
            raise ValueError(f"Target year {target_year} not found in dataset year columns.")

        target_col = self.year_map[target_year]
        non_null_year_counts = self.df[list(self.year_map.values())].notna().sum(axis=1)

        missing_pct = (self.df.isna().mean() * 100).sort_values(ascending=False)

        profile = {
            "dataset_path": str(self.config.dataset_path),
            "row_count": int(self.df.shape[0]),
            "column_count": int(self.df.shape[1]),
            "year_start": int(min(self.year_map)),
            "year_end": int(max(self.year_map)),
            "year_count": int(len(self.year_map)),
            "target_year": int(target_year),
            "target_column": target_col,
            "target_non_null": int(self.df[target_col].notna().sum()),
            "target_missing": int(self.df[target_col].isna().sum()),
            "non_null_years_per_country": {
                "min": int(non_null_year_counts.min()),
                "max": int(non_null_year_counts.max()),
                "mean": float(non_null_year_counts.mean()),
                "median": float(non_null_year_counts.median()),
            },
            "top_missing_columns_pct": {
                col: float(val) for col, val in missing_pct.head(12).items()
            },
        }

        safe_json_dump(self.config.output_dir / "dataset_profile.json", profile)
        self.results["dataset_profile"] = profile
        return profile

    def _build_cross_sectional_frame(
        self,
        target_year: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, List[str], List[str], List[int]]:
        if target_year is None:
            target_year = self.config.target_year or max(self.year_map)

        feature_years = [y for y in sorted(self.year_map) if y < target_year]
        if len(feature_years) < self.config.max_lag:
            raise ValueError("Not enough historical years to build cross-sectional features.")

        mmr_feature_cols = [self.year_map[y] for y in feature_years]
        target_col = self.year_map[target_year]

        numeric_cols = list(self.numeric_meta_cols) + mmr_feature_cols
        categorical_cols = list(self.categorical_meta_cols)

        keep_cols = [c for c in ["Country"] + categorical_cols + numeric_cols + [target_col] if c in self.df.columns]
        frame = self.df[keep_cols].copy()
        frame = frame.rename(columns={target_col: "target"})
        frame = frame.dropna(subset=["target"]).reset_index(drop=True)

        return frame, numeric_cols, categorical_cols, feature_years

    def run_linear_regression_module(self) -> Dict[str, Any]:
        frame, numeric_cols, categorical_cols, feature_years = self._build_cross_sectional_frame()

        X = frame[numeric_cols + categorical_cols]
        y = frame["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=True,
        )

        model = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)),
                ("model", LinearRegression()),
            ]
        )

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_metrics = regression_metrics(y_train, train_pred)
        test_metrics = regression_metrics(y_test, test_pred)

        preprocessor = model.named_steps["preprocess"]
        regressor = model.named_steps["model"]

        feature_names = preprocessor.get_feature_names_out()
        coefficient_series = pd.Series(regressor.coef_, index=feature_names)
        coefficient_series = coefficient_series.reindex(
            coefficient_series.abs().sort_values(ascending=False).index
        )

        if self.config.make_plots:
            plot_actual_vs_pred(
                y_true=y_test,
                y_pred=test_pred,
                title="Linear Regression - Actual vs Predicted",
                output_path=self.config.output_dir / "linear_regression_actual_vs_pred.png",
            )
            plot_top_features(
                feature_scores=coefficient_series.abs(),
                title="Linear Regression - Top Absolute Coefficients",
                output_path=self.config.output_dir / "linear_regression_top_coefficients.png",
                top_n=20,
            )

        output = {
            "module": "linear_regression",
            "target_year": int(self.config.target_year or max(self.year_map)),
            "feature_year_start": int(min(feature_years)),
            "feature_year_end": int(max(feature_years)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "metrics": {
                "train": train_metrics,
                "test": test_metrics,
            },
            "top_absolute_coefficients": {
                key: float(value) for key, value in coefficient_series.head(30).items()
            },
        }

        safe_json_dump(self.config.output_dir / "linear_regression_results.json", output)
        self.results["linear_regression"] = output
        return output

    def _build_horizon_training_frame(self) -> pd.DataFrame:
        years = sorted(self.year_map)
        min_year = min(years)
        max_year = max(years)

        rows: List[Dict[str, Any]] = []

        for _, row in self.df.iterrows():
            country = row.get("Country", "Unknown")

            base_meta: Dict[str, Any] = {"Country": country}
            for col in self.categorical_meta_cols + self.numeric_meta_cols:
                base_meta[col] = row.get(col, np.nan)

            series = {year: row[self.year_map[year]] for year in years}

            for anchor_year in years:
                if anchor_year - (self.config.max_lag - 1) < min_year:
                    continue

                lag_values: List[float] = []
                valid_lag_block = True
                for lag in range(0, self.config.max_lag):
                    year_at_lag = anchor_year - lag
                    value = series.get(year_at_lag)
                    if pd.isna(value):
                        valid_lag_block = False
                        break
                    lag_values.append(float(value))

                if not valid_lag_block:
                    continue

                common_features: Dict[str, Any] = {
                    **base_meta,
                    "anchor_year": int(anchor_year),
                    "lag_1": lag_values[0],
                    "lag_2": lag_values[1],
                    "lag_3": lag_values[2],
                    "lag_4": lag_values[3],
                    "lag_5": lag_values[4],
                    "rolling_mean_3": float(np.mean(lag_values[:3])),
                    "rolling_mean_5": float(np.mean(lag_values[:5])),
                    "trend_3": float(lag_values[0] - lag_values[2]),
                    "trend_5": float(lag_values[0] - lag_values[4]),
                }

                for horizon in self.config.horizons:
                    target_year = anchor_year + horizon
                    if target_year > max_year:
                        continue

                    target_value = series.get(target_year)
                    if pd.isna(target_value):
                        continue

                    rows.append(
                        {
                            **common_features,
                            "horizon": int(horizon),
                            "target_year": int(target_year),
                            "target": float(target_value),
                        }
                    )

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def _build_latest_anchor_rows(self) -> pd.DataFrame:
        years = sorted(self.year_map)
        max_year = max(years)

        rows: List[Dict[str, Any]] = []
        for _, row in self.df.iterrows():
            country = row.get("Country", "Unknown")

            lag_values: List[float] = []
            valid = True
            for lag in range(0, self.config.max_lag):
                year_at_lag = max_year - lag
                value = row[self.year_map[year_at_lag]]
                if pd.isna(value):
                    valid = False
                    break
                lag_values.append(float(value))

            if not valid:
                continue

            data: Dict[str, Any] = {
                "Country": country,
                "anchor_year": int(max_year),
                "lag_1": lag_values[0],
                "lag_2": lag_values[1],
                "lag_3": lag_values[2],
                "lag_4": lag_values[3],
                "lag_5": lag_values[4],
                "rolling_mean_3": float(np.mean(lag_values[:3])),
                "rolling_mean_5": float(np.mean(lag_values[:5])),
                "trend_3": float(lag_values[0] - lag_values[2]),
                "trend_5": float(lag_values[0] - lag_values[4]),
            }

            for col in self.categorical_meta_cols + self.numeric_meta_cols:
                data[col] = row.get(col, np.nan)

            rows.append(data)

        return pd.DataFrame(rows)

    def run_random_forest_module(self) -> Dict[str, Any]:
        frame, numeric_cols, categorical_cols, _ = self._build_cross_sectional_frame()
        X = frame[numeric_cols + categorical_cols]
        y = frame["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=True,
        )

        rf_regressor = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=700,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                        min_samples_leaf=1,
                        min_samples_split=2,
                    ),
                ),
            ]
        )

        rf_regressor.fit(X_train, y_train)
        train_pred = rf_regressor.predict(X_train)
        test_pred = rf_regressor.predict(X_test)

        reg_train_metrics = regression_metrics(y_train, train_pred)
        reg_test_metrics = regression_metrics(y_test, test_pred)

        q1, q2 = y.quantile([0.33, 0.66]).tolist()
        risk_labels = pd.cut(
            y,
            bins=[-np.inf, q1, q2, np.inf],
            labels=["Low Risk", "Mid Risk", "High Risk"],
        )

        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X,
            risk_labels,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=True,
            stratify=risk_labels,
        )

        rf_classifier = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        rf_classifier.fit(X_train_c, y_train_c)
        class_pred = rf_classifier.predict(X_test_c)

        class_accuracy = float(accuracy_score(y_test_c, class_pred))
        class_report = classification_report(y_test_c, class_pred, output_dict=True, zero_division=0)

        preprocessor = rf_regressor.named_steps["preprocess"]
        reg_model = rf_regressor.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        importance_series = pd.Series(reg_model.feature_importances_, index=feature_names)
        importance_series = importance_series.sort_values(ascending=False)

        if self.config.make_plots:
            plot_actual_vs_pred(
                y_true=y_test,
                y_pred=test_pred,
                title="Random Forest Regressor - Actual vs Predicted",
                output_path=self.config.output_dir / "random_forest_actual_vs_pred.png",
            )
            plot_top_features(
                feature_scores=importance_series,
                title="Random Forest - Top Feature Importances",
                output_path=self.config.output_dir / "random_forest_feature_importance.png",
                top_n=20,
            )

        # Multi-horizon random forest forecasting block.
        horizon_frame = self._build_horizon_training_frame()
        horizon_reports: Dict[int, Dict[str, Any]] = {}

        future_anchor_rows = self._build_latest_anchor_rows()
        future_predictions_rows: List[Dict[str, Any]] = []

        if not horizon_frame.empty:
            horizon_numeric_cols = [
                "anchor_year",
                "lag_1",
                "lag_2",
                "lag_3",
                "lag_4",
                "lag_5",
                "rolling_mean_3",
                "rolling_mean_5",
                "trend_3",
                "trend_5",
            ] + self.numeric_meta_cols
            horizon_categorical_cols = list(self.categorical_meta_cols)
            horizon_feature_cols = horizon_numeric_cols + horizon_categorical_cols

            for horizon in sorted(horizon_frame["horizon"].unique()):
                subset = horizon_frame[horizon_frame["horizon"] == horizon].copy()
                subset = subset.dropna(subset=["target"])

                if len(subset) < 50:
                    horizon_reports[int(horizon)] = {
                        "status": "skipped",
                        "reason": "insufficient_samples",
                        "sample_count": int(len(subset)),
                    }
                    continue

                cutoff_year = int(np.floor(subset["anchor_year"].quantile(1 - self.config.test_size)))
                train_block = subset[subset["anchor_year"] <= cutoff_year]
                test_block = subset[subset["anchor_year"] > cutoff_year]

                if train_block.empty or test_block.empty:
                    train_block, test_block = train_test_split(
                        subset,
                        test_size=self.config.test_size,
                        random_state=self.config.random_state,
                        shuffle=True,
                    )

                X_train_h = train_block[horizon_feature_cols]
                y_train_h = train_block["target"]
                X_test_h = test_block[horizon_feature_cols]
                y_test_h = test_block["target"]

                rf_h = Pipeline(
                    steps=[
                        (
                            "preprocess",
                            build_preprocessor(
                                numeric_cols=horizon_numeric_cols,
                                categorical_cols=horizon_categorical_cols,
                                scale_numeric=False,
                            ),
                        ),
                        (
                            "model",
                            RandomForestRegressor(
                                n_estimators=600,
                                random_state=self.config.random_state,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                )

                rf_h.fit(X_train_h, y_train_h)
                h_pred = rf_h.predict(X_test_h)
                h_metrics = regression_metrics(y_test_h, h_pred)

                # Estimate a pragmatic interval from individual tree predictions.
                transformed_test = rf_h.named_steps["preprocess"].transform(X_test_h)
                tree_predictions_test = np.column_stack(
                    [tree.predict(transformed_test) for tree in rf_h.named_steps["model"].estimators_]
                )
                lower_test = np.percentile(tree_predictions_test, 5, axis=1)
                upper_test = np.percentile(tree_predictions_test, 95, axis=1)
                interval_coverage = float(np.mean((y_test_h.to_numpy() >= lower_test) & (y_test_h.to_numpy() <= upper_test)))
                interval_width = float(np.mean(upper_test - lower_test))

                horizon_reports[int(horizon)] = {
                    "status": "trained",
                    "sample_count": int(len(subset)),
                    "train_count": int(len(train_block)),
                    "test_count": int(len(test_block)),
                    "anchor_cutoff_year": int(cutoff_year),
                    "metrics": h_metrics,
                    "interval_coverage_5_95": interval_coverage,
                    "mean_interval_width_5_95": interval_width,
                }

                if not future_anchor_rows.empty:
                    X_future = future_anchor_rows[horizon_feature_cols]
                    future_pred = rf_h.predict(X_future)

                    transformed_future = rf_h.named_steps["preprocess"].transform(X_future)
                    tree_predictions_future = np.column_stack(
                        [tree.predict(transformed_future) for tree in rf_h.named_steps["model"].estimators_]
                    )
                    lower_future = np.percentile(tree_predictions_future, 5, axis=1)
                    upper_future = np.percentile(tree_predictions_future, 95, axis=1)

                    max_observed_year = max(self.year_map)
                    for i in range(len(future_anchor_rows)):
                        future_predictions_rows.append(
                            {
                                "Country": future_anchor_rows.iloc[i]["Country"],
                                "horizon": int(horizon),
                                "forecast_year": int(max_observed_year + int(horizon)),
                                "prediction": float(future_pred[i]),
                                "lower_5": float(lower_future[i]),
                                "upper_95": float(upper_future[i]),
                            }
                        )

        if future_predictions_rows:
            future_df = pd.DataFrame(future_predictions_rows)
            future_df.to_csv(
                self.config.output_dir / "random_forest_multi_horizon_forecasts.csv",
                index=False,
            )

        output = {
            "module": "random_forest",
            "cross_sectional_regression": {
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "metrics": {
                    "train": reg_train_metrics,
                    "test": reg_test_metrics,
                },
                "top_feature_importances": {
                    key: float(value) for key, value in importance_series.head(30).items()
                },
            },
            "risk_classifier": {
                "class_thresholds": {"q33": float(q1), "q66": float(q2)},
                "class_distribution": {k: int(v) for k, v in risk_labels.value_counts().items()},
                "test_accuracy": class_accuracy,
                "classification_report": class_report,
            },
            "multi_horizon_forecasting": horizon_reports,
        }

        safe_json_dump(self.config.output_dir / "random_forest_results.json", output)
        self.results["random_forest"] = output
        return output

    def _build_long_frame(self) -> pd.DataFrame:
        year_columns = [self.year_map[y] for y in sorted(self.year_map)]

        id_columns = [
            col
            for col in ["Country"] + self.categorical_meta_cols + self.numeric_meta_cols
            if col in self.df.columns
        ]

        long_df = self.df[id_columns + year_columns].melt(
            id_vars=id_columns,
            value_vars=year_columns,
            var_name="year_column",
            value_name="mmr",
        )

        long_df["year"] = long_df["year_column"].str.extract(r"(\d{4})").astype(int)
        long_df["mmr"] = pd.to_numeric(long_df["mmr"], errors="coerce")
        long_df = long_df.sort_values(["Country", "year"]).reset_index(drop=True)

        long_df["lag_1"] = long_df.groupby("Country")["mmr"].shift(1)
        long_df["lag_2"] = long_df.groupby("Country")["mmr"].shift(2)
        long_df["lag_3"] = long_df.groupby("Country")["mmr"].shift(3)
        long_df["rolling_mean_5"] = long_df.groupby("Country")["mmr"].transform(
            lambda s: s.rolling(window=5, min_periods=3).mean()
        )
        long_df["mmr_change"] = long_df.groupby("Country")["mmr"].diff()

        year_min = long_df["year"].min()
        year_max = long_df["year"].max()
        year_scale = max(1, int(year_max - year_min))
        long_df["year_norm"] = (long_df["year"] - year_min) / year_scale

        long_df["target_next_mmr"] = long_df.groupby("Country")["mmr"].shift(-1)
        long_df["target_year"] = long_df["year"] + 1

        return long_df

    def run_xgboost_module(self) -> Dict[str, Any]:
        try:
            from xgboost import XGBRegressor
        except Exception as ex:
            output = {
                "module": "xgboost",
                "status": "skipped",
                "reason": "xgboost_not_available",
                "detail": str(ex),
            }
            safe_json_dump(self.config.output_dir / "xgboost_results.json", output)
            self.results["xgboost"] = output
            return output

        long_df = self._build_long_frame()

        numeric_cols = [
            "year_norm",
            "lag_1",
            "lag_2",
            "lag_3",
            "rolling_mean_5",
            "mmr_change",
        ] + self.numeric_meta_cols
        categorical_cols = list(self.categorical_meta_cols)

        model_df = long_df.dropna(subset=numeric_cols + ["target_next_mmr"]).copy()

        if model_df.empty:
            output = {
                "module": "xgboost",
                "status": "skipped",
                "reason": "no_rows_after_feature_engineering",
            }
            safe_json_dump(self.config.output_dir / "xgboost_results.json", output)
            self.results["xgboost"] = output
            return output

        max_target_year = int(model_df["target_year"].max())
        split_target_year = max_target_year - self.config.holdout_years

        train_df = model_df[model_df["target_year"] <= split_target_year]
        test_df = model_df[model_df["target_year"] > split_target_year]

        if train_df.empty or test_df.empty:
            train_df, test_df = train_test_split(
                model_df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                shuffle=True,
            )
            split_target_year = -1

        feature_cols = numeric_cols + categorical_cols

        X_train = train_df[feature_cols]
        y_train = train_df["target_next_mmr"]
        X_test = test_df[feature_cols]
        y_test = test_df["target_next_mmr"]

        xgb_pipeline = Pipeline(
            steps=[
                (
                    "preprocess",
                    build_preprocessor(
                        numeric_cols=numeric_cols,
                        categorical_cols=categorical_cols,
                        scale_numeric=False,
                    ),
                ),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=700,
                        learning_rate=0.03,
                        max_depth=6,
                        min_child_weight=2,
                        subsample=0.85,
                        colsample_bytree=0.8,
                        reg_alpha=0.05,
                        reg_lambda=1.2,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
            ]
        )

        xgb_pipeline.fit(X_train, y_train)

        train_pred = xgb_pipeline.predict(X_train)
        test_pred = xgb_pipeline.predict(X_test)

        train_metrics = regression_metrics(y_train, train_pred)
        test_metrics = regression_metrics(y_test, test_pred)

        feature_names = xgb_pipeline.named_steps["preprocess"].get_feature_names_out()
        importances = xgb_pipeline.named_steps["model"].feature_importances_
        importance_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        if self.config.make_plots:
            plot_actual_vs_pred(
                y_true=y_test,
                y_pred=test_pred,
                title="XGBoost - Actual vs Predicted (Next-Year MMR)",
                output_path=self.config.output_dir / "xgboost_actual_vs_pred.png",
            )
            plot_top_features(
                feature_scores=importance_series,
                title="XGBoost - Top Feature Importances",
                output_path=self.config.output_dir / "xgboost_feature_importance.png",
                top_n=20,
            )

        # Next-year forecast from each country's latest observed row.
        latest_year = int(max(self.year_map))
        latest_rows = model_df[model_df["year"] == latest_year].copy()

        if latest_rows.empty:
            latest_rows = model_df.sort_values("year").groupby("Country", as_index=False).tail(1)

        future_forecasts = latest_rows[["Country"]].copy()
        future_forecasts["forecast_year"] = latest_year + 1
        future_forecasts["xgboost_prediction"] = xgb_pipeline.predict(latest_rows[feature_cols])
        future_forecasts = future_forecasts.sort_values("xgboost_prediction", ascending=False)
        future_forecasts.to_csv(self.config.output_dir / "xgboost_next_year_forecasts.csv", index=False)

        output = {
            "module": "xgboost",
            "status": "trained",
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "temporal_split_target_year_cutoff": int(split_target_year),
            "metrics": {
                "train": train_metrics,
                "test": test_metrics,
            },
            "top_feature_importances": {
                key: float(value) for key, value in importance_series.head(30).items()
            },
            "next_year_forecast_country_count": int(len(future_forecasts)),
        }

        safe_json_dump(self.config.output_dir / "xgboost_results.json", output)
        self.results["xgboost"] = output
        return output

    def _select_arima_countries(self) -> List[str]:
        if self.config.arima_country:
            if "Country" not in self.df.columns:
                return []
            matches = self.df[self.df["Country"].str.lower() == self.config.arima_country.lower()]
            if matches.empty:
                return []
            return [matches.iloc[0]["Country"]]

        latest_col = self.year_map[max(self.year_map)]
        ranked = self.df[["Country", latest_col]].copy()
        ranked = ranked.dropna(subset=[latest_col])
        ranked = ranked.sort_values(latest_col, ascending=False)
        return ranked["Country"].head(self.config.arima_top_countries).tolist()

    def _extract_country_series(self, country: str) -> pd.Series:
        row = self.df[self.df["Country"] == country]
        if row.empty:
            return pd.Series(dtype=float)

        row_values = row.iloc[0]
        points: Dict[int, float] = {}
        for year in sorted(self.year_map):
            value = row_values[self.year_map[year]]
            if pd.notna(value):
                points[year] = float(value)

        if not points:
            return pd.Series(dtype=float)

        years = sorted(points.keys())
        values = [points[y] for y in years]
        idx = pd.PeriodIndex(years, freq="Y").to_timestamp()
        return pd.Series(values, index=idx, name=country)

    def run_arima_module(self) -> Dict[str, Any]:
        candidate_countries = self._select_arima_countries()

        forecasts_rows: List[Dict[str, Any]] = []
        backtest_scores: List[Dict[str, float]] = []
        failures: List[Dict[str, str]] = []

        for country in candidate_countries:
            series = self._extract_country_series(country)
            if series.empty or len(series) < 12:
                failures.append({"country": country, "reason": "insufficient_history"})
                continue

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    model = ARIMA(series, order=self.config.arima_order)
                    fitted = model.fit()

                forecast_obj = fitted.get_forecast(steps=self.config.arima_forecast_horizon)
                forecast_mean = forecast_obj.predicted_mean
                conf_int = forecast_obj.conf_int(alpha=0.10)

                for i in range(self.config.arima_forecast_horizon):
                    ts = forecast_mean.index[i]
                    forecasts_rows.append(
                        {
                            "Country": country,
                            "forecast_year": int(ts.year),
                            "prediction": float(forecast_mean.iloc[i]),
                            "lower_90": float(conf_int.iloc[i, 0]),
                            "upper_90": float(conf_int.iloc[i, 1]),
                        }
                    )

                # Small holdout backtest using the last forecast_horizon points.
                if len(series) > self.config.arima_forecast_horizon + 8:
                    train = series.iloc[:-self.config.arima_forecast_horizon]
                    test = series.iloc[-self.config.arima_forecast_horizon :]

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        bt_model = ARIMA(train, order=self.config.arima_order).fit()

                    bt_pred = bt_model.forecast(steps=self.config.arima_forecast_horizon)
                    bt_metrics = regression_metrics(test, bt_pred)
                    bt_metrics["country"] = country
                    backtest_scores.append(bt_metrics)

                if self.config.make_plots:
                    plt.figure(figsize=(10, 5))
                    plt.plot(series.index.year, series.values, label="Historical")
                    plt.plot(forecast_mean.index.year, forecast_mean.values, label="Forecast", linestyle="--")
                    plt.fill_between(
                        forecast_mean.index.year,
                        conf_int.iloc[:, 0].values,
                        conf_int.iloc[:, 1].values,
                        alpha=0.2,
                        label="90% CI",
                    )
                    plt.title(f"ARIMA Forecast - {country}")
                    plt.xlabel("Year")
                    plt.ylabel("Maternal Mortality Ratio")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(self.config.output_dir / f"arima_forecast_{country.replace(' ', '_')}.png", dpi=140)
                    plt.close()

            except Exception as ex:
                failures.append({"country": country, "reason": str(ex)})

        forecast_df = pd.DataFrame(forecasts_rows)
        if not forecast_df.empty:
            forecast_df.to_csv(self.config.output_dir / "arima_forecasts.csv", index=False)

        if backtest_scores:
            backtest_df = pd.DataFrame(backtest_scores)
            summary_scores = {
                "mean_mae": float(backtest_df["mae"].mean()),
                "mean_rmse": float(backtest_df["rmse"].mean()),
                "mean_r2": float(backtest_df["r2"].replace([np.inf, -np.inf], np.nan).dropna().mean())
                if "r2" in backtest_df.columns
                else float("nan"),
                "country_count": int(backtest_df["country"].nunique()),
            }
            backtest_df.to_csv(self.config.output_dir / "arima_backtest_scores.csv", index=False)
        else:
            summary_scores = {
                "mean_mae": float("nan"),
                "mean_rmse": float("nan"),
                "mean_r2": float("nan"),
                "country_count": 0,
            }

        output = {
            "module": "arima",
            "order": {
                "p": int(self.config.arima_order[0]),
                "d": int(self.config.arima_order[1]),
                "q": int(self.config.arima_order[2]),
            },
            "forecast_horizon_years": int(self.config.arima_forecast_horizon),
            "countries_selected": candidate_countries,
            "countries_successful": sorted(forecast_df["Country"].unique().tolist()) if not forecast_df.empty else [],
            "failures": failures,
            "backtest_summary": summary_scores,
            "forecast_rows": int(len(forecast_df)),
        }

        safe_json_dump(self.config.output_dir / "arima_results.json", output)
        self.results["arima"] = output
        return output

    def _create_combined_score_table(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        lr = self.results.get("linear_regression", {})
        if lr.get("metrics", {}).get("test"):
            rows.append(
                {
                    "model": "Linear Regression",
                    **lr["metrics"]["test"],
                }
            )

        rf = self.results.get("random_forest", {})
        rf_test = rf.get("cross_sectional_regression", {}).get("metrics", {}).get("test")
        if rf_test:
            rows.append(
                {
                    "model": "Random Forest Regressor",
                    **rf_test,
                }
            )

        xgb = self.results.get("xgboost", {})
        xgb_test = xgb.get("metrics", {}).get("test")
        if xgb_test:
            rows.append(
                {
                    "model": "XGBoost Regressor",
                    **xgb_test,
                }
            )

        arima = self.results.get("arima", {})
        arima_bt = arima.get("backtest_summary", {})
        if arima_bt and arima_bt.get("country_count", 0) > 0:
            rows.append(
                {
                    "model": "ARIMA (avg backtest)",
                    "mae": arima_bt.get("mean_mae"),
                    "rmse": arima_bt.get("mean_rmse"),
                    "r2": arima_bt.get("mean_r2"),
                    "mse": np.nan,
                }
            )

        table = pd.DataFrame(rows)
        if not table.empty:
            table = table[[col for col in ["model", "mae", "mse", "rmse", "r2"] if col in table.columns]]
            table.to_csv(self.config.output_dir / "combined_model_comparison.csv", index=False)

        return table

    def _write_markdown_summary(self, comparison: pd.DataFrame) -> None:
        lines: List[str] = []
        lines.append("# Unified Maternal Mortality Pipeline Report")
        lines.append("")

        profile = self.results.get("dataset_profile", {})
        lines.append("## Dataset Profile")
        lines.append(f"- Rows: {profile.get('row_count', 'N/A')}")
        lines.append(f"- Columns: {profile.get('column_count', 'N/A')}")
        lines.append(
            f"- Year Range: {profile.get('year_start', 'N/A')} to {profile.get('year_end', 'N/A')}"
        )
        lines.append(f"- Target Year: {profile.get('target_year', 'N/A')}")
        lines.append("")

        lines.append("## Modules Executed")
        for key in ["linear_regression", "random_forest", "xgboost", "arima"]:
            section = self.results.get(key, {})
            status = section.get("status", "trained")
            lines.append(f"- {key}: {status}")
        lines.append("")

        lines.append("## Model Comparison")
        if comparison.empty:
            lines.append("No model comparison available.")
        else:
            lines.append("")
            lines.append(comparison.to_markdown(index=False))
        lines.append("")

        lines.append("## Output Artifacts")
        lines.append(f"- Directory: {self.config.output_dir}")
        lines.append("- JSON results for each module")
        lines.append("- CSV forecasts and model comparison")
        lines.append("- PNG visualizations (if plotting enabled)")

        report_path = self.config.output_dir / "unified_pipeline_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")

    def run_all(self) -> Dict[str, Any]:
        self.load_and_profile_dataset()

        self.run_linear_regression_module()
        self.run_random_forest_module()
        self.run_xgboost_module()
        self.run_arima_module()

        comparison = self._create_combined_score_table()
        self._write_markdown_summary(comparison)

        safe_json_dump(self.config.output_dir / "unified_results.json", self.results)
        return self.results


def parse_args() -> UnifiedConfig:
    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Unified Maternal Mortality Prediction Pipeline using LR, RF, XGBoost, and ARIMA."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(project_root / "Maternal_Mortality.csv"),
        help="Path to maternal mortality CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "outputs_unified_pipeline"),
        help="Directory where reports, metrics, and plots will be saved.",
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=None,
        help="Target year for cross-sectional models (default: latest available year).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size fraction for cross-sectional and fallback splits.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="Random Forest multi-horizon forecast steps.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=5,
        help="Maximum lag features for horizon forecasting.",
    )
    parser.add_argument(
        "--holdout-years",
        type=int,
        default=4,
        help="Number of last target years reserved for XGBoost temporal test split.",
    )
    parser.add_argument(
        "--arima-order",
        type=int,
        nargs=3,
        default=[1, 1, 1],
        help="ARIMA order p d q.",
    )
    parser.add_argument(
        "--arima-horizon",
        type=int,
        default=5,
        help="ARIMA forecast horizon in years.",
    )
    parser.add_argument(
        "--arima-top-countries",
        type=int,
        default=12,
        help="Top countries by latest MMR used for ARIMA when no specific country is provided.",
    )
    parser.add_argument(
        "--arima-country",
        type=str,
        default=None,
        help="Optional specific country for ARIMA forecast.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation.",
    )

    args = parser.parse_args()

    return UnifiedConfig(
        dataset_path=Path(args.data).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        target_year=args.target_year,
        test_size=args.test_size,
        random_state=args.random_state,
        horizons=tuple(args.horizons),
        max_lag=args.max_lag,
        holdout_years=args.holdout_years,
        arima_order=(args.arima_order[0], args.arima_order[1], args.arima_order[2]),
        arima_forecast_horizon=args.arima_horizon,
        arima_top_countries=args.arima_top_countries,
        arima_country=args.arima_country,
        make_plots=not args.no_plots,
    )


def main() -> None:
    config = parse_args()
    pipeline = UnifiedMaternalMortalityPipeline(config)
    results = pipeline.run_all()

    print("Unified pipeline completed.")
    print(f"Output directory: {config.output_dir}")
    print("Generated module keys:")
    for key in results.keys():
        print(f"- {key}")


if __name__ == "__main__":
    main()
