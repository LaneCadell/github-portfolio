"""
Primary Quantile Regression Model
LightGBM-based AVM generating Q_10, Q_50, Q_90 price forecasts
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrimaryQuantileAVM:
    """
    LightGBM Quantile Regression for property valuation.
    Generates 10th, 50th, and 90th percentile price predictions.
    """

    def __init__(
        self,
        quantiles: list = None,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 8,
        seed: int = 42,
    ):
        """
        Initialize Primary Quantile AVM.

        Args:
            quantiles: List of quantiles to predict (default: [0.1, 0.5, 0.9])
            n_estimators: Number of LightGBM boosting rounds
            learning_rate: Learning rate
            max_depth: Tree depth
            seed: Random seed
        """
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.seed = seed
        self.models = {}  # {quantile: trained_model}
        self.feature_names = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        categorical_features: Optional[list] = None,
        verbose: int = 0,
    ) -> None:
        """
        Train quantile regression models for each quantile.

        Args:
            X_train: Training features
            y_train: Training prices (target)
            categorical_features: List of categorical feature names
            verbose: LightGBM verbosity (0 = silent)
        """
        feature_cols = [col for col in X_train.columns if col != "id"]
        self.feature_names = feature_cols

        logger.info(f"Training Primary AVM on {len(X_train)} samples, {len(feature_cols)} features")

        for quantile in self.quantiles:
            logger.info(f"  Training Q_{int(quantile*100)} model...")

            params = {
                "objective": "quantile",
                "alpha": quantile,
                "metric": "quantile",
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "num_leaves": 2 ** self.max_depth,
                "min_data_in_leaf": 50,
                "verbose": verbose,
                "seed": self.seed,
                "bagging_fraction": 0.8,
                "feature_fraction": 0.8,
                "lambda_l2": 1.0,
            }

            train_data = lgb.Dataset(
                X_train[feature_cols],
                label=y_train,
                categorical_feature=categorical_features or [],
                free_raw_data=False,
            )

            self.models[quantile] = lgb.train(
                params,
                train_data,
                num_boost_round=self.n_estimators,
                verbose_eval=False,
            )

        logger.info("Primary AVM training complete")

    def predict(self, X_test: pd.DataFrame) -> Dict[float, np.ndarray]:
        """
        Generate quantile predictions.

        Args:
            X_test: Test features

        Returns:
            Dictionary mapping quantile -> predictions
        """
        feature_cols = [col for col in X_test.columns if col != "id"]
        
        predictions = {}
        for quantile, model in self.models.items():
            pred = model.predict(X_test[feature_cols])
            predictions[quantile] = np.array(pred)

        return predictions

    def predict_with_uncertainty(
        self, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty bands: Q_10, Q_50, Q_90.

        Args:
            X_test: Test features

        Returns:
            Tuple of (Q_10, Q_50, Q_90) predictions
        """
        preds = self.predict(X_test)
        return preds[0.1], preds[0.5], preds[0.9]

    def get_feature_importance(self, quantile: float = 0.5, top_n: int = 15) -> pd.DataFrame:
        """
        Extract feature importance for a specific quantile model.

        Args:
            quantile: Quantile model to extract importance from (default: 0.5)
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance ranking
        """
        if quantile not in self.models:
            logger.warning(f"Model for Q_{int(quantile*100)} not found")
            return pd.DataFrame()

        model = self.models[quantile]
        importance = model.feature_importance(importance_type="gain")

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)

        return importance_df.head(top_n)

    def calculate_raw_errors(
        self, X_val: pd.DataFrame, y_val: pd.Series
    ) -> np.ndarray:
        """
        Calculate out-of-fold absolute percentage errors for meta-model training.
        Uses Q_50 predictions as point estimates.

        Args:
            X_val: Validation features
            y_val: Validation prices (actual)

        Returns:
            Array of absolute percentage errors
        """
        q50_preds = self.predict(X_val)[0.5]
        ape = np.abs((y_val.values - q50_preds) / y_val.values)
        return ape

    def calculate_interval_coverage(
        self, X_val: pd.DataFrame, y_val: pd.Series
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate empirical coverage of the [Q_10, Q_90] interval.

        Args:
            X_val: Validation features
            y_val: Validation prices

        Returns:
            Tuple of (coverage_percentage, coverage_mask)
        """
        q10, _, q90 = self.predict_with_uncertainty(X_val)
        
        coverage_mask = (y_val.values >= q10) & (y_val.values <= q90)
        coverage = coverage_mask.mean() * 100

        logger.info(f"Empirical coverage of [Q_10, Q_90]: {coverage:.2f}%")
        return coverage, coverage_mask


if __name__ == "__main__":
    from data_loader import RealEstateDataLoader
    from feature_engineering import FeatureEngineer

    # Load and prepare data
    loader = RealEstateDataLoader("/Users/kayleighinman/Downloads/kc_house_data.csv")
    df = loader.load_data()
    train, val, test = loader.temporal_split()

    # Engineer features
    engineer = FeatureEngineer()
    train_features = engineer.engineer_features(train, fit=True)
    val_features = engineer.engineer_features(val, fit=False)

    # Train primary model
    model = PrimaryQuantileAVM()
    model.train(train_features, train["price"])

    # Evaluate on validation
    q10, q50, q90 = model.predict_with_uncertainty(val_features)
    print(f"Validation Q_50 MAPE: {np.mean(np.abs((val['price'].values - q50) / val['price'].values)) * 100:.2f}%")

    # Calculate coverage
    coverage, _ = model.calculate_interval_coverage(val_features, val["price"])

    # Feature importance
    print("\nTop 10 features by importance:")
    print(model.get_feature_importance(quantile=0.5, top_n=10))
