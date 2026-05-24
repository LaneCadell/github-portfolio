"""
Feature Engineering Module
Structural attributes, macroeconomic signals, and target encoding for Kings County data
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Construct features for primary and secondary models"""

    def __init__(self):
        """Initialize feature engineer"""
        self.feature_names = []
        self.zipcode_encoder = {}
        self.is_fitted = False

    def engineer_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Engineer structural, temporal, and macro features.

        Args:
            df: Kings County housing DataFrame
            fit: If True, fit encoders on this data; otherwise use existing

        Returns:
            Feature DataFrame ready for modeling
        """
        columns_to_remove = ["lat", "long", "sqft_living15", "sqft_lot15"]
        df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
        features = df[["id"]].copy()

        # --- STRUCTURAL FEATURES ---
        
        # Square footage (primary driver of value)
        features["sqft_living"] = df["sqft_living"]
        features["sqft_lot"] = df["sqft_lot"]
        features["sqft_ratio"] = df["sqft_living"] / (df["sqft_lot"] + 1)

        # Layout & quality
        features["bedrooms"] = df["bedrooms"]
        features["bathrooms"] = df["bathrooms"]
        features["room_density"] = df["bedrooms"] / (df["bathrooms"] + 0.5)  # rooms per bath
        
        # Grade (quality indicator, 1-13 scale)
        features["grade"] = df["grade"]
        features["grade_normalized"] = (df["grade"] - df["grade"].min()) / (df["grade"].max() - df["grade"].min())

        # Condition (1-5 scale)
        features["condition"] = df["condition"]

        # Age and renovation status
        features["age"] = 2015 - df["yr_built"]  # Approximate current year
        features["is_renovated"] = (df["yr_renovated"] > 0).astype(int)
        features["renovation_age"] = df.apply(
            lambda x: 2015 - x["yr_renovated"] if x["yr_renovated"] > 0 else np.nan,
            axis=1
        )
        features["renovation_age"] = features["renovation_age"].fillna(features["age"])

        # Above-ground features
        features["sqft_above"] = df["sqft_above"]
        features["sqft_basement"] = df["sqft_living"] - df["sqft_above"]
        features["basement_ratio"] = features["sqft_basement"] / (df["sqft_living"] + 1)

        # Optional macroeconomic features
        if "mortgage_rate" in df.columns:
            features["mortgage_rate"] = df["mortgage_rate"]

        # Special features
        features["has_view"] = df["view"].astype(bool).astype(int)
        features["has_waterfront"] = df["waterfront"].astype(int)
        features["has_view_or_waterfront"] = ((df["view"] > 0) | (df["waterfront"] > 0)).astype(int)

        # Floor count
        features["floors"] = df["floors"]

        # --- TEMPORAL FEATURES ---
        df_date = pd.to_datetime(df["date"])
        features["sale_year"] = df_date.dt.year
        features["sale_month"] = df_date.dt.month
        features["sale_quarter"] = df_date.dt.quarter
        features["sale_day_of_year"] = df_date.dt.dayofyear
        
        # Seasonality encoding
        features["is_spring"] = (features["sale_month"].isin([3, 4, 5])).astype(int)
        features["is_summer"] = (features["sale_month"].isin([6, 7, 8])).astype(int)
        features["is_fall"] = (features["sale_month"].isin([9, 10, 11])).astype(int)

        # --- GEOGRAPHIC FEATURES (Zipcode-based) ---
        
        # Dummy encode top zips, leave rare as "other"
        zip_counts = df["zipcode"].value_counts()
        top_zips = zip_counts[zip_counts >= 50].index.tolist()
        
        features["zipcode_encoded"] = df["zipcode"].apply(
            lambda x: x if x in top_zips else 99999
        )

        # Safe target encoding on training set only
        if fit:
            self.zipcode_encoder = {}
            for zip_code in features["zipcode_encoded"].unique():
                mask = features["zipcode_encoded"] == zip_code
                median_price = df[mask]["price"].median()
                self.zipcode_encoder[zip_code] = median_price
            self.is_fitted = True
            logger.info(f"Fitted target encoder for {len(self.zipcode_encoder)} zipcodes")

        elif not self.is_fitted:
            raise ValueError("Must fit encoder first. Call with fit=True on training data.")

        features["zipcode_target_encoding"] = features["zipcode_encoded"].map(
            self.zipcode_encoder
        ).fillna(df["price"].median())

        # --- INTERACTION FEATURES ---
        features["luxury_large"] = (
            (features["grade_normalized"] > 0.6) & (features["sqft_living"] > 3000)
        ).astype(int)
        
        features["age_condition_interaction"] = features["age"] * features["condition"]
        
        features["waterfront_premium"] = features["has_waterfront"] * features["sqft_living"]

        # --- DERIVED QUALITY METRICS ---
        features["property_score"] = (
            features["grade_normalized"] * 0.4 +
            features["condition"] / 5.0 * 0.3 +
            (1 - features["age"] / features["age"].max()) * 0.3
        )

        self.feature_names = [
            col for col in features.columns if col != "id"
        ]
        logger.info(f"Engineered {len(self.feature_names)} features")

        return features

    def get_feature_names(self) -> list:
        """Return list of engineered feature names"""
        return self.feature_names

    @staticmethod
    def normalize_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Standardize features using training set statistics.

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Tuple of (normalized X_train, normalized X_test)
        """
        from sklearn.preprocessing import StandardScaler

        feature_cols = [col for col in X_train.columns if col != "id"]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[feature_cols])
        X_test_scaled = scaler.transform(X_test[feature_cols])

        X_train_norm = X_train.copy()
        X_test_norm = X_test.copy()
        
        X_train_norm[feature_cols] = X_train_scaled
        X_test_norm[feature_cols] = X_test_scaled

        return X_train_norm, X_test_norm


if __name__ == "__main__":
    import os
    from data_loader import RealEstateDataLoader

    data_path = os.path.join(os.path.dirname(__file__), "kc_house_data.csv")
    loader = RealEstateDataLoader(data_path)
    df = loader.load_data()

    # Create temporal split
    train, val, test = loader.temporal_split()

    # Engineer features
    engineer = FeatureEngineer()
    train_features = engineer.engineer_features(train, fit=True)
    val_features = engineer.engineer_features(val, fit=False)
    test_features = engineer.engineer_features(test, fit=False)

    print("Training features shape:", train_features.shape)
    print("Features:", engineer.get_feature_names()[:10])
    print("\nSample features:")
    print(train_features.head())
