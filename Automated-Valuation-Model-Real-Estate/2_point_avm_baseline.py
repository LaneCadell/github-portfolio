"""
Naive baseline comparison and point-based AVM lift evaluation.
"""

import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from data_loader import RealEstateDataLoader
from feature_engineering import FeatureEngineer
from primary_model import PrimaryQuantileAVM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "2_point_avm_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_output_path(name: str) -> Path:
    path = OUTPUT_DIR / name
    return path


def prepare_chronological_data(data_path: str) -> tuple:
    raw_df = pd.read_csv(data_path)
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    loader = RealEstateDataLoader(data_path)
    filtered_df, _ = loader.apply_outlier_pipeline(raw_df)
    loader.df = filtered_df
    train_df, val_df, test_df = loader.temporal_split(train_frac=0.7, val_frac=0.15)

    impute_columns = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "sqft_above", "sqft_basement"]
    loader.fit_imputation(train_df, feature_columns=impute_columns)
    train_df = loader.transform_imputation(train_df, feature_columns=impute_columns)
    val_df = loader.transform_imputation(val_df, feature_columns=impute_columns)
    test_df = loader.transform_imputation(test_df, feature_columns=impute_columns)

    return train_df, val_df, test_df


class RollingMedianBaseline:
    """Naive reference model using a rolling median price from training data."""

    def __init__(self):
        self.monthly_medians = None
        self.global_median = None

    def fit(self, train_df: pd.DataFrame) -> None:
        train_df = train_df.copy()
        train_df["year_month"] = train_df["date"].dt.to_period("M")
        self.monthly_medians = train_df.groupby("year_month")["price"].median().to_dict()
        self.global_median = train_df["price"].median()
        logger.info("Fitted rolling median baseline from training data")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df["year_month"] = df["date"].dt.to_period("M")

        def median_for_row(row):
            if row["year_month"] in self.monthly_medians:
                return self.monthly_medians[row["year_month"]]
            return self.global_median

        return df.apply(median_for_row, axis=1).values


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ape = np.abs((y_true - y_pred) / (y_true + 1e-10))
    metrics = {
        "MAPE": float(np.mean(ape)) * 100.0,
        "MdAPE": float(np.median(ape)) * 100.0,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    return metrics


def plot_lift_comparison(y_true: np.ndarray, point_preds: np.ndarray, baseline_preds: np.ndarray) -> None:
    errors_model = np.abs((y_true - point_preds) / (y_true + 1e-10))
    errors_baseline = np.abs((y_true - baseline_preds) / (y_true + 1e-10))
    median_model = float(np.median(errors_model))
    median_baseline = float(np.median(errors_baseline))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(errors_model, label="Q50 AVM", fill=True, color="#1b76d1", alpha=0.6, ax=ax)
    sns.kdeplot(errors_baseline, label="Baseline Median", fill=True, color="#d1495b", alpha=0.6, ax=ax)
    ax.axvline(median_model, color="#1b76d1", linestyle="--", linewidth=1.5,
               label=f"Q50 AVM median MAPE: {median_model * 100:.2f}%")
    ax.axvline(median_baseline, color="#d1495b", linestyle="--", linewidth=1.5,
               label=f"Baseline median MAPE: {median_baseline * 100:.2f}%")
    ax.set_title("AVM Point Forecast Lift: Absolute Percentage Error Density")
    ax.set_xlabel("Absolute Percentage Error")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = create_output_path("model_lift_comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved model lift comparison plot to {path}")


def save_lift_markdown(primary_metrics: dict, baseline_metrics: dict) -> None:
    output_path = create_output_path("baseline_lift_summary.md")
    lines = [
        "# AVM Point Forecast vs. Naive Baseline",
        "",
        "## Lift Metrics",
        "",
        "| Metric | Q50 AVM | Rolling Median Baseline | Improvement |",
        "|---|---:|---:|---:|",
    ]

    for key in ["MAPE", "MdAPE", "RMSE"]:
        primary_value = primary_metrics[key]
        baseline_value = baseline_metrics[key]
        improvement = baseline_value - primary_value
        lines.append(
            f"| {key} | {primary_value:.2f} | {baseline_value:.2f} | {improvement:.2f} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- Baseline predictions are produced by a rolling monthly median price model fitted only on the training partition.",
        "- All error calculations are strictly evaluated on the held-out test set to prevent leakage.",
        "- Improvement is computed as Baseline minus Q50 AVM so positive values indicate model lift.",
    ])

    output_path.write_text("\n".join(lines))
    logger.info(f"Saved baseline lift summary to {output_path}")


def run_baseline_comparison(data_path: str) -> dict:
    train_df, val_df, test_df = prepare_chronological_data(data_path)

    feature_engineer = FeatureEngineer()
    train_features = feature_engineer.engineer_features(train_df, fit=True)
    test_features = feature_engineer.engineer_features(test_df, fit=False)

    primary = PrimaryQuantileAVM(quantiles=[0.1, 0.5, 0.9], n_estimators=500, learning_rate=0.05, max_depth=8)
    primary.train(train_features, train_df["price"])

    q50_test = primary.predict(test_features)[0.5]

    baseline = RollingMedianBaseline()
    baseline.fit(train_df)
    baseline_preds = baseline.predict(test_df)

    primary_metrics = compute_metrics(test_df["price"].values, q50_test)
    baseline_metrics = compute_metrics(test_df["price"].values, baseline_preds)

    plot_lift_comparison(test_df["price"].values, q50_test, baseline_preds)
    save_lift_markdown(primary_metrics, baseline_metrics)

    logger.info("Baseline comparison complete")
    return {
        "primary_metrics": primary_metrics,
        "baseline_metrics": baseline_metrics,
        "lift_plot": str(create_output_path("model_lift_comparison.png")),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare the AVM Q50 point forecast against a rolling median baseline.")
    parser.add_argument("--data-path", required=True, help="Path to kc_house_data.csv")
    args = parser.parse_args()

    results = run_baseline_comparison(args.data_path)
    logger.info("Final metrics:")
    logger.info(results)
