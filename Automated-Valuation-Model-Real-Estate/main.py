"""
Main Orchestration Script
End-to-end pipeline: data loading → feature engineering → dual-model training →
calibration → offer generation → market feedback integration.
"""

import argparse
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime

from data_loader import RealEstateDataLoader
from feature_engineering import FeatureEngineer
from primary_model import PrimaryQuantileAVM
from meta_error_model import MetaErrorPredictor
from calibration_engine import CalibrationEngine
from offer_engine import OfferEngine
from market_feedback import MarketFeedbackLoop

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RiskAwareREPricingPipeline:
    """
    Production-ready dual-model real estate pricing engine.
    Integrates primary quantile AVM, meta-error predictor, calibration, and market feedback.
    """

    def __init__(self, data_path: str):
        """
        Initialize pipeline.

        Args:
            data_path: Path to Kings County housing dataset
        """
        self.data_path = data_path
        self.data_loader = None
        self.feature_engineer = None
        self.primary_model = None
        self.meta_error_model = None
        self.calibrator = None
        self.offer_engine = None
        self.market_feedback = None

    def load_and_split_data(self) -> tuple:
        """
        Load and create chronological train/val/test splits.

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("="*60)
        logger.info("STAGE 0: Data Loading")
        logger.info("="*60)

        self.data_loader = RealEstateDataLoader(self.data_path)
        df = self.data_loader.load_data()

        df, outlier_report = self.data_loader.apply_outlier_pipeline(df)
        logger.info("Applied deterministic outlier pipeline to the loaded dataset")
        logger.info(
            "Outlier report: " + ", ".join(
                [f"{k}={v}" for k, v in outlier_report.items() if k != "original_count" and k != "final_count"]
            )
        )

        self.data_loader.df = df
        train_df, val_df, test_df = self.data_loader.temporal_split(
            train_frac=0.7, val_frac=0.15
        )

        feature_columns = [
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "sqft_lot",
            "sqft_above",
            "sqft_basement",
        ]
        train_df, val_df, test_df = self.data_loader.preprocess_split_sets(
            train_df, val_df, test_df, feature_columns=feature_columns
        )

        return train_df, val_df, test_df

    def engineer_features(self, train_df, val_df, test_df) -> tuple:
        """
        Engineer features for both primary and meta-error models.

        Returns:
            Tuple of (train_features, val_features, test_features)
        """
        logger.info("="*60)
        logger.info("STAGE 1: Feature Engineering")
        logger.info("="*60)

        self.feature_engineer = FeatureEngineer()

        train_features = self.feature_engineer.engineer_features(train_df, fit=True)
        val_features = self.feature_engineer.engineer_features(val_df, fit=False)
        test_features = self.feature_engineer.engineer_features(test_df, fit=False)

        # Normalize
        train_feat_norm, val_feat_norm = FeatureEngineer.normalize_features(
            train_features, val_features
        )
        _, test_feat_norm = FeatureEngineer.normalize_features(
            train_features, test_features
        )

        return train_feat_norm, val_feat_norm, test_feat_norm

    def train_primary_model(self, train_features, train_df) -> PrimaryQuantileAVM:
        """
        Train LightGBM quantile regression model.

        Returns:
            Trained PrimaryQuantileAVM
        """
        logger.info("="*60)
        logger.info("STAGE 2: Primary Quantile AVM Training")
        logger.info("="*60)

        self.primary_model = PrimaryQuantileAVM(
            quantiles=[0.1, 0.5, 0.9],
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
        )

        self.primary_model.train(train_features, train_df["price"])

        # Feature importance
        logger.info("Top 10 features (Primary Model, Q_50):")
        importance = self.primary_model.get_feature_importance(quantile=0.5, top_n=10)
        for idx, row in importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.0f}")

        return self.primary_model

    def train_meta_error_model(self, train_features, train_df) -> MetaErrorPredictor:
        """
        Train meta-error predictor on primary model's absolute percentage errors.

        Returns:
            Trained MetaErrorPredictor
        """
        logger.info("="*60)
        logger.info("STAGE 3: Meta-Error Predictor Training")
        logger.info("="*60)

        # Calculate errors on training set
        ape_errors = self.primary_model.calculate_raw_errors(train_features, train_df["price"])

        self.meta_error_model = MetaErrorPredictor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
        )

        self.meta_error_model.train(train_features, ape_errors)

        # Feature importance
        logger.info("Top 10 error drivers (Meta-Error Model):")
        importance = self.meta_error_model.get_feature_importance(top_n=10)
        for idx, row in importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.0f}")

        return self.meta_error_model

    def calibrate_intervals(
        self, val_features, val_df, target_coverage: float = 0.80
    ) -> CalibrationEngine:
        """
        Learn interval scaling factors for empirical coverage.

        Args:
            val_features: Validation features
            val_df: Validation data
            target_coverage: Target empirical coverage (default: 0.80)

        Returns:
            Calibrated CalibrationEngine
        """
        logger.info("="*60)
        logger.info("STAGE 4: Calibration Layer")
        logger.info("="*60)

        q10_raw, q50_raw, q90_raw = self.primary_model.predict_with_uncertainty(val_features)

        self.calibrator = CalibrationEngine(target_coverage=target_coverage)
        self.calibrator.calibrate(q10_raw, q50_raw, q90_raw, val_df["price"].values)

        return self.calibrator

    def generate_offers(
        self,
        test_features,
        test_df,
        base_spread: float = 0.05,
        alpha_uncertainty: float = 0.15,
    ) -> pd.DataFrame:
        """
        Generate buy-side offers with uncertainty penalties and ensemble weights.

        Args:
            test_features: Test features
            test_df: Test data
            base_spread: Base bid spread
            alpha_uncertainty: Uncertainty penalty weight

        Returns:
            DataFrame with pricing report
        """
        logger.info("="*60)
        logger.info("STAGE 5: Offer Engine & Risk Scoring")
        logger.info("="*60)

        # Generate raw predictions
        q10_raw, q50_raw, q90_raw = self.primary_model.predict_with_uncertainty(test_features)

        # Calibrate
        q10_cal, q50_cal, q90_cal = self.calibrator.apply_calibration(
            q10_raw, q50_raw, q90_raw
        )

        # Generate EES scores
        ees_scores = self.meta_error_model.predict_ees(test_features)

        # Create offer engine
        self.offer_engine = OfferEngine(
            base_spread=base_spread,
            alpha_uncertainty=alpha_uncertainty,
        )

        # Calculate uncertainty and generate offers
        uncertainty_score = self.offer_engine.calculate_uncertainty_score(
            q10_cal, q50_cal, q90_cal
        )
        buy_offers = self.offer_engine.generate_buy_offer(
            q50_cal, uncertainty_score, ees_scores
        )
        ensemble_weights = self.offer_engine.calculate_ensemble_weights(ees_scores)

        # Generate report
        report = self.offer_engine.generate_pricing_report(
            test_df["id"].values,
            q10_cal, q50_cal, q90_cal,
            uncertainty_score,
            ees_scores,
            buy_offers,
            ensemble_weights,
        )

        logger.info(f"\nGenerated {len(report)} property offers")
        logger.info(f"Mean buy offer: ${report['buy_offer_price'].mean():,.0f}")
        logger.info(f"Mean ensemble weight: {report['ensemble_weight'].mean():.4f}")
        logger.info(f"Risk tier distribution:\n{report['risk_tier'].value_counts()}")

        return report

    def run_full_pipeline(self) -> Dict:
        """
        Execute complete pipeline from data loading to offer generation.

        Returns:
            Dictionary with all results and models
        """
        # Load and split
        train_df, val_df, test_df = self.load_and_split_data()

        # Engineer features
        train_feat, val_feat, test_feat = self.engineer_features(train_df, val_df, test_df)

        # Train models
        self.train_primary_model(train_feat, train_df)
        self.train_meta_error_model(train_feat, train_df)

        # Calibrate
        self.calibrate_intervals(val_feat, val_df, target_coverage=0.80)

        # Generate offers
        offer_report = self.generate_offers(test_feat, test_df)

        logger.info("="*60)
        logger.info("Pipeline Complete")
        logger.info("="*60)

        return {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "train_features": train_feat,
            "val_features": val_feat,
            "test_features": test_feat,
            "primary_model": self.primary_model,
            "meta_error_model": self.meta_error_model,
            "calibrator": self.calibrator,
            "offer_engine": self.offer_engine,
            "offer_report": offer_report,
        }


def cli_main():
    """Command-line interface for the pipeline"""

    parser = argparse.ArgumentParser(
        description="Risk-Aware Real Estate Pricing Engine"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to Kings County housing dataset (CSV)",
    )
    parser.add_argument(
        "--base-spread",
        type=float,
        default=0.05,
        help="Base bid spread (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=0.80,
        help="Target empirical coverage for calibration (default: 0.80 = 80%%)",
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = RiskAwareREPricingPipeline(args.data_path)
    results = pipeline.run_full_pipeline()

    # Display results
    offer_report = results["offer_report"]
    print("\n" + "="*80)
    print("TOP 10 HIGHEST CONFIDENCE OFFERS (Lowest Risk)")
    print("="*80)
    print(offer_report.head(10)[
        ["property_id", "median_forecast", "buy_offer_price", "ees_score", "risk_tier"]
    ].to_string(index=False))

    print("\n" + "="*80)
    print("PORTFOLIO SUMMARY")
    print("="*80)
    print(f"Total properties priced: {len(offer_report)}")
    print(f"Total portfolio value (median forecast): ${offer_report['median_forecast'].sum():,.0f}")
    print(f"Total buy offer value: ${offer_report['buy_offer_price'].sum():,.0f}")
    print(f"Average bid spread: {offer_report['spread_vs_median'].mean():.2f}%")


if __name__ == "__main__":
    cli_main()
