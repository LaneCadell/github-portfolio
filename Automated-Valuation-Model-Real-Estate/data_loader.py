"""
Data Loading & Exploration Module
Kings County Housing Dataset preparation and validation
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealEstateDataLoader:
    """Load and validate Kings County housing data"""

    def __init__(self, filepath: str):
        """
        Initialize data loader.

        Args:
            filepath: Path to Kings County housing dataset (CSV)
        """
        self.filepath = filepath
        self.df = None
        self.metadata = {}

    def load_data(self) -> pd.DataFrame:
        """
        Load and perform initial validation.

        Returns:
            Loaded DataFrame with basic cleaning
        """
        logger.info(f"Loading data from {self.filepath}...")
        self.df = pd.read_csv(self.filepath)

        logger.info(f"Shape: {self.df.shape}")
        logger.info(f"Columns: {self.df.columns.tolist()}")
        logger.info(f"Missing values:\n{self.df.isnull().sum()}")

        # Store metadata
        self.metadata = {
            "total_records": len(self.df),
            "load_timestamp": datetime.now(),
            "price_range": (self.df["price"].min(), self.df["price"].max()),
            "date_range": (self.df["date"].min(), self.df["date"].max()),
        }

        # Remove rows with missing target
        if self.df["price"].isnull().any():
            logger.warning(f"Removing {self.df['price'].isnull().sum()} rows with missing price")
            self.df = self.df.dropna(subset=["price"])

        # Basic outlier removal (price sanity checks)
        price_q1 = self.df["price"].quantile(0.01)
        price_q99 = self.df["price"].quantile(0.99)
        logger.info(f"Price range (1st-99th percentile): ${price_q1:,.0f} - ${price_q99:,.0f}")

        # Remove extreme outliers (likely data errors)
        self.df = self.df[(self.df["price"] >= price_q1) & (self.df["price"] <= price_q99)]
        logger.info(f"After outlier removal: {len(self.df)} records")

        return self.df

    def temporal_split(
        self, train_frac: float = 0.7, val_frac: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create chronological train/val/test splits to prevent lookahead bias.

        Args:
            train_frac: Fraction for training (default: 0.7)
            val_frac: Fraction for validation (default: 0.15)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.df is None:
            raise ValueError("Load data first with load_data()")

        # Sort by date chronologically
        df_sorted = self.df.sort_values("date").reset_index(drop=True)

        n = len(df_sorted)
        train_idx = int(n * train_frac)
        val_idx = int(n * (train_frac + val_frac))

        train_df = df_sorted.iloc[:train_idx].copy()
        val_df = df_sorted.iloc[train_idx:val_idx].copy()
        test_df = df_sorted.iloc[val_idx:].copy()

        logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        logger.info(
            f"Train dates: {train_df['date'].min()} to {train_df['date'].max()}"
        )
        logger.info(
            f"Val dates: {val_df['date'].min()} to {val_df['date'].max()}"
        )
        logger.info(
            f"Test dates: {test_df['date'].min()} to {test_df['date'].max()}"
        )

        return train_df, val_df, test_df

    def rolling_chronological_split(
        self, train_window: int = 5000, val_window: int = 1000, step: int = 1000
    ) -> list:
        """
        Create rolling chronological folds for training secondary models without leakage.

        Args:
            train_window: Number of samples for training each fold
            val_window: Number of samples for validation each fold
            step: Step size to roll forward

        Returns:
            List of tuples (train_df, val_df) for each fold
        """
        if self.df is None:
            raise ValueError("Load data first with load_data()")

        df_sorted = self.df.sort_values("date").reset_index(drop=True)
        folds = []

        for i in range(0, len(df_sorted) - train_window - val_window, step):
            train_start = i
            train_end = i + train_window
            val_start = train_end
            val_end = val_start + val_window

            if val_end > len(df_sorted):
                break

            train_fold = df_sorted.iloc[train_start:train_end].copy()
            val_fold = df_sorted.iloc[val_start:val_end].copy()

            folds.append((train_fold, val_fold))

        logger.info(f"Created {len(folds)} rolling chronological folds")
        return folds

    @staticmethod
    def validate_feature_distributions(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
        """
        Validate that train and test feature distributions are reasonable.

        Args:
            X_train: Training feature set
            X_test: Test feature set
        """
        logger.info("Feature distribution validation:")
        for col in X_train.columns:
            train_mean = X_train[col].mean()
            test_mean = X_test[col].mean()
            drift = abs(test_mean - train_mean) / (abs(train_mean) + 1e-6)
            status = "⚠️ DRIFT" if drift > 0.15 else "✓"
            logger.info(f"  {col}: train_μ={train_mean:.2f}, test_μ={test_mean:.2f} ({drift:.2%}) {status}")


if __name__ == "__main__":
    loader = RealEstateDataLoader("/Users/kayleighinman/Downloads/kc_house_data.csv")
    df = loader.load_data()
    print(df.head())
    print("\nMetadata:", loader.metadata)

    # Test temporal split
    train, val, test = loader.temporal_split()
    print(f"\nTemporal split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Test rolling folds
    folds = loader.rolling_chronological_split(train_window=5000, val_window=1000)
    print(f"Rolling folds: {len(folds)} folds")
