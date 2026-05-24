"""
Exploratory Data Analysis and Deterministic Outlier Pipeline
Phase 1 for the Risk-Aware Home Pricing Engine.
"""

import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_loader import RealEstateDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "1_eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load raw King's County data without hidden filtering."""
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    loader = RealEstateDataLoader(data_path)
    return loader.augment_with_macro_features(df)


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add features useful for EDA and outlier identification."""
    df = df.copy()
    df["price_per_sqft"] = df["price"] / (df["sqft_living"] + 1e-10)
    df["age"] = 2015 - df["yr_built"].fillna(2015)
    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    return df


def save_figure(fig, filename: str) -> None:
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved plot {path}")


def plot_distribution(df: pd.DataFrame, column: str, title: str, xlabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[column].dropna(), kde=True, stat="density", ax=ax, color="#2a5d84")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    save_figure(fig, filename)


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df.sample(min(len(df), 5000), random_state=42), x=x_col, y=y_col, alpha=0.35, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    save_figure(fig, filename)


def plot_macro_trend(df: pd.DataFrame) -> None:
    if "mortgage_rate" not in df.columns:
        logger.info("No mortgage_rate column present; skipping macro trend plot.")
        return

    monthly = df.groupby("sale_month")["mortgage_rate"].mean().reset_index()
    monthly["sale_month"] = pd.to_datetime(monthly["sale_month"])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=monthly, x="sale_month", y="mortgage_rate", ax=ax, color="#2a5d84")
    ax.set_title("Average Monthly Mortgage Rate by Sale Month")
    ax.set_xlabel("Sale Month")
    ax.set_ylabel("30-Year Mortgage Rate (%)")
    save_figure(fig, "mortgage_rate_trend.png")


def save_data_quality_report(report: dict, imputation_report: dict, before_count: int, after_count: int) -> None:
    report_path = OUTPUT_DIR / "data_quality_report.md"
    lines = [
        "# Data Quality Report",
        "",
        "## Outlier Pipeline Summary",
        "",
        f"- Original dataset size: **{before_count:,}** rows",
        f"- Final dataset size after deterministic outlier filtering: **{after_count:,}** rows",
        "",
        "### Records removed by filter type",
        "",
        "| Filter | Rows removed |",
        "|---|---:|",
    ]

    for filter_name, count in report.items():
        if filter_name == "final_count":
            continue
        lines.append(f"| {filter_name.replace('_', ' ').capitalize()} | {count:,} |")

    lines.extend([
        "",
        "## Imputation Frequency by Feature",
        "",
        "The imputation frequencies below are calculated using training-set statistics only, ensuring zero leakage into validation/test partitions.",
        "",
        "| Feature | Train missing | Val missing | Test missing |",
        "|---|---:|---:|---:|",
    ])

    for feature, counts in imputation_report.items():
        lines.append(
            f"| {feature} | {counts['train']:,} | {counts['val']:,} | {counts['test']:,} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- Imputation uses group-wise median values computed from the training partition only.",
        "- All missing architectural and geographic values are filled using the most conservative zipcode-level statistics available.",
        "- The final dataset is ready for chronological train/validation/test splitting without leakage.",
    ])

    report_path.write_text("\n".join(lines))
    logger.info(f"Saved data quality report to {report_path}")


def build_imputation_report(
    loader: RealEstateDataLoader,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list,
) -> dict:
    imputation_report = {}
    loader.fit_imputation(train_df, feature_columns=feature_columns)

    for feature in feature_columns:
        imputation_report[feature] = {
            "train": int(train_df[feature].isna().sum()),
            "val": int(val_df[feature].isna().sum()),
            "test": int(test_df[feature].isna().sum()),
        }

    return imputation_report


def run_eda(data_path: str) -> dict:
    raw_df = load_raw_data(data_path)
    raw_df = add_derived_fields(raw_df)

    plot_distribution(raw_df, "price", "Price Distribution", "Sale Price (USD)", "price_distribution.png")
    plot_distribution(raw_df, "price_per_sqft", "Price per Square Foot Distribution", "Price / SqFt (USD)", "price_per_sqft_distribution.png")
    plot_distribution(raw_df, "sqft_living", "Living Area Distribution", "Living Area (sqft)", "sqft_living_distribution.png")
    plot_distribution(raw_df, "age", "Approximate Age Distribution", "Age (years)", "age_distribution.png")
    plot_scatter(raw_df, "sqft_living", "price", "Price versus Living Area", "price_vs_sqft.png")
    plot_scatter(raw_df, "age", "price_per_sqft", "Age versus Price per Square Foot", "age_vs_ppsqft.png")
    plot_macro_trend(raw_df)

    loader = RealEstateDataLoader(data_path)
    filtered_df, outlier_report = loader.apply_outlier_pipeline(raw_df)
    before_count = len(raw_df)
    after_count = len(filtered_df)

    logger.info("Outlier pipeline complete. Building train/validation/test splits...")
    loader.df = filtered_df
    train_df, val_df, test_df = loader.temporal_split(train_frac=0.7, val_frac=0.15)

    imputation_columns = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "sqft_above", "sqft_basement"]
    imputation_report = build_imputation_report(loader, train_df, val_df, test_df, imputation_columns)
    save_data_quality_report(outlier_report, imputation_report, before_count, after_count)

    return {
        "raw_count": before_count,
        "filtered_count": after_count,
        "outlier_report": outlier_report,
        "imputation_report": imputation_report,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EDA and outlier preprocessing for the King County dataset.")
    parser.add_argument("--data-path", required=True, help="Path to kc_house_data.csv")
    args = parser.parse_args()

    run_eda(args.data_path)
