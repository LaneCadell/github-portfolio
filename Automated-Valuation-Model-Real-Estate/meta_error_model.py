"""
Meta-Error Predictor Layer
Trains secondary model on primary model's out-of-fold errors.
Outputs Expected Error Score (EES) mapped to 0-100 scale.
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaErrorPredictor:
    """
    Secondary regression model trained on primary AVM's absolute percentage errors.
    Generates Expected Error Score (0-100) for dynamic ensembling and risk weighting.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        seed: int = 42,
    ):
        """
        Initialize Meta-Error Predictor.

        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Tree depth
            seed: Random seed
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.seed = seed
        self.model = None
        self.feature_names = None
        
        # Calibration stats for EES mapping
        self.error_min = None
        self.error_max = None
        self.error_median = None
        self.error_std = None

    def train(
        self,
        X_train: pd.DataFrame,
        error_targets: np.ndarray,
        categorical_features: Optional[list] = None,
        verbose: int = 0,
    ) -> None:
        """
        Train meta-error model on primary model's absolute percentage errors.

        Args:
            X_train: Same features as primary model
            error_targets: Array of absolute percentage errors (from primary model's Q_50)
            categorical_features: List of categorical feature names
            verbose: LightGBM verbosity
        """
        feature_cols = [col for col in X_train.columns if col != "id"]
        self.feature_names = feature_cols

        # Store error statistics for EES calibration
        self.error_min = np.min(error_targets)
        self.error_max = np.max(error_targets)
        self.error_median = np.median(error_targets)
        self.error_std = np.std(error_targets)

        logger.info(f"Training Meta-Error Predictor on {len(X_train)} samples")
        logger.info(
            f"  Error stats: min={self.error_min:.4f}, max={self.error_max:.4f}, "
            f"median={self.error_median:.4f}, std={self.error_std:.4f}"
        )

        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": 2 ** self.max_depth,
            "min_data_in_leaf": 30,
            "verbose": verbose,
            "seed": self.seed,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.8,
            "lambda_l2": 1.0,
        }

        train_data = lgb.Dataset(
            X_train[feature_cols],
            label=error_targets,
            categorical_feature=categorical_features or [],
            free_raw_data=False,
        )

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            verbose_eval=False,
        )

        logger.info("Meta-Error Predictor training complete")

    def predict_raw_errors(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict raw absolute percentage errors.

        Args:
            X_test: Test features

        Returns:
            Array of predicted absolute percentage errors
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        feature_cols = [col for col in X_test.columns if col != "id"]
        predictions = self.model.predict(X_test[feature_cols])
        
        # Clip to realistic bounds
        predictions = np.clip(predictions, self.error_min, self.error_max)
        return predictions

    def map_to_ees(self, predicted_errors: np.ndarray) -> np.ndarray:
        """
        Map predicted errors to Expected Error Score (0-100).
        
        Lower predicted error -> Higher EES (more confident prediction)
        Higher predicted error -> Lower EES (less confident prediction)

        Args:
            predicted_errors: Array of predicted absolute percentage errors

        Returns:
            Array of EES scores (0-100 scale)
        """
        # Min-max scaling: error -> confidence
        # error_min -> 100 (most confident), error_max -> 0 (least confident)
        ees = 100.0 * (self.error_max - predicted_errors) / (self.error_max - self.error_min + 1e-10)
        ees = np.clip(ees, 0, 100)
        
        return ees

    def predict_ees(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generate Expected Error Score (0-100) directly.

        Args:
            X_test: Test features

        Returns:
            Array of EES scores (0-100)
        """
        predicted_errors = self.predict_raw_errors(X_test)
        ees = self.map_to_ees(predicted_errors)
        return ees

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Extract feature importance from meta-error model.

        Args:
            top_n: Number of top features

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            logger.warning("Model not trained yet")
            return pd.DataFrame()

        importance = self.model.feature_importance(importance_type="gain")

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)

        return importance_df.head(top_n)

    @staticmethod
    def train_meta_error_from_cv_folds(
        folds: list,
        feature_engineer,
        primary_model_class,
        primary_model_kwargs: dict = None,
        meta_model_kwargs: dict = None,
    ) -> Tuple["MetaErrorPredictor", np.ndarray, np.ndarray]:
        """
        Train meta-error model using chronological cross-validation folds.
        Prevents data leakage by training secondary model on out-of-fold errors.

        Args:
            folds: List of (train_df, val_df) tuples from rolling_chronological_split
            feature_engineer: Fitted FeatureEngineer instance
            primary_model_class: Class of primary AVM model
            primary_model_kwargs: Kwargs for primary model
            meta_model_kwargs: Kwargs for meta-error model

        Returns:
            Tuple of (meta_model, oof_errors, oof_indices)
        """
        primary_model_kwargs = primary_model_kwargs or {}
        meta_model_kwargs = meta_model_kwargs or {}

        all_errors = []
        all_features = []
        all_indices = []

        logger.info(f"Training meta-error model on {len(folds)} CV folds...")

        for fold_idx, (train_fold, val_fold) in enumerate(folds):
            logger.info(f"  Fold {fold_idx + 1}/{len(folds)}")

            # Engineer features on this fold
            train_feat = feature_engineer.engineer_features(train_fold, fit=False)
            val_feat = feature_engineer.engineer_features(val_fold, fit=False)

            # Train primary model
            primary = primary_model_class(**primary_model_kwargs)
            primary.train(train_feat, train_fold["price"])

            # Calculate out-of-fold errors
            oof_errors = primary.calculate_raw_errors(val_feat, val_fold["price"])

            all_errors.append(oof_errors)
            all_features.append(val_feat)
            all_indices.append(val_fold.index)

        # Combine all out-of-fold data
        X_meta = pd.concat(all_features, ignore_index=False)
        y_meta_errors = np.concatenate(all_errors)
        oof_indices = pd.Index(np.concatenate(all_indices))

        logger.info(f"Training meta-error model on {len(X_meta)} out-of-fold samples")

        # Train meta-error model
        meta_model = MetaErrorPredictor(**meta_model_kwargs)
        meta_model.train(X_meta, y_meta_errors)

        return meta_model, y_meta_errors, oof_indices


if __name__ == "__main__":
    import os
    from data_loader import RealEstateDataLoader
    from feature_engineering import FeatureEngineer
    from primary_model import PrimaryQuantileAVM

    data_path = os.path.join(os.path.dirname(__file__), "kc_house_data.csv")
    loader = RealEstateDataLoader(data_path)
    df = loader.load_data()
    train, val, test = loader.temporal_split()

    # Engineer features
    engineer = FeatureEngineer()
    train_features = engineer.engineer_features(train, fit=True)
    val_features = engineer.engineer_features(val, fit=False)

    # Train primary model
    primary = PrimaryQuantileAVM()
    primary.train(train_features, train["price"])

    # Calculate errors for meta-model training
    errors = primary.calculate_raw_errors(train_features, train["price"])

    # Train meta-error model
    meta = MetaErrorPredictor()
    meta.train(train_features, errors)

    # Generate EES on validation
    ees_scores = meta.predict_ees(val_features)
    print(f"EES scores - min: {ees_scores.min():.2f}, max: {ees_scores.max():.2f}, median: {np.median(ees_scores):.2f}")

    print("\nTop 10 features for error prediction:")
    print(meta.get_feature_importance(top_n=10))
