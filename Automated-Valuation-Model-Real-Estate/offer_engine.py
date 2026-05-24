"""
Offer Engine & Ensembling Framework
Constructs asymmetric buy-side offers with dynamic uncertainty penalties.
Provides ensemble weight calculations for multi-model integration.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OfferEngine:
    """
    Generates buy-side pricing with dynamic risk-adjusted margins.
    Integrates calibrated quantiles, expected error scores, and ensembling weights.
    """

    def __init__(
        self,
        base_spread: float = 0.05,
        alpha_uncertainty: float = 0.15,
        beta_ees: float = 0.001,
        max_penalty: float = 0.25,
    ):
        """
        Initialize Offer Engine.

        Args:
            base_spread: Base bid-ask spread (default: 5%)
            alpha_uncertainty: Weight on uncertainty score (default: 0.15)
            beta_ees: Weight on EES penalty (default: 0.001)
            max_penalty: Maximum penalty cap (default: 25%)
        """
        self.base_spread = base_spread
        self.alpha_uncertainty = alpha_uncertainty
        self.beta_ees = beta_ees
        self.max_penalty = max_penalty

    def calculate_uncertainty_score(
        self,
        q10_cal: np.ndarray,
        q50: np.ndarray,
        q90_cal: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Uncertainty Score: normalized interval width relative to median.

        $$\text{Uncertainty Score} = \frac{Q_{90\_cal} - Q_{10\_cal}}{Q_{50}}$$

        Args:
            q10_cal: Calibrated 10th percentile
            q50: 50th percentile (median)
            q90_cal: Calibrated 90th percentile

        Returns:
            Array of uncertainty scores (0-1 range)
        """
        interval_widths = q90_cal - q10_cal
        uncertainty_score = interval_widths / (q50 + 1e-10)
        return uncertainty_score

    def calculate_uncertainty_penalty(
        self,
        uncertainty_score: np.ndarray,
        ees_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Combine uncertainty and error scores into asymmetric buy-side penalty.

        $$\text{Penalty} = \min(\alpha \cdot \text{Uncertainty} + \beta \cdot \text{EES}, \, \text{max\_penalty})$$

        Args:
            uncertainty_score: Uncertainty scores (from calculate_uncertainty_score)
            ees_scores: Expected Error Scores (0-100 from meta-error model)

        Returns:
            Array of penalties (0-max_penalty)
        """
        # Convert EES (0-100) to penalty impact (0-1)
        # Higher EES -> lower penalty (more confident)
        ees_penalty_term = (100 - ees_scores) / 100.0

        penalty = (
            self.alpha_uncertainty * uncertainty_score +
            self.beta_ees * ees_penalty_term
        )

        penalty = np.clip(penalty, 0, self.max_penalty)
        return penalty

    def generate_buy_offer(
        self,
        q50: np.ndarray,
        uncertainty_score: np.ndarray,
        ees_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Generate risk-adjusted buy-side offer prices.

        $$\text{Buy Offer} = Q_{50} \times (1 - \text{base\_spread} - \text{Penalty})$$

        Args:
            q50: 50th percentile forecasts (median)
            uncertainty_score: Uncertainty scores
            ees_scores: Expected Error Scores (0-100)

        Returns:
            Array of buy-side offer prices
        """
        penalty = self.calculate_uncertainty_penalty(uncertainty_score, ees_scores)
        buy_offer = q50 * (1 - self.base_spread - penalty)

        return buy_offer

    def calculate_ensemble_weights(
        self,
        ees_scores: np.ndarray,
        epsilon: float = 1.0,
    ) -> np.ndarray:
        """
        Generate dynamic ensemble weights based on expected error scores.
        
        Use inverse-error variance for downstream model integration:
        $$W_{\text{AVM}} \propto \frac{1}{\epsilon + \text{EES}}$$

        Args:
            ees_scores: Expected Error Scores (0-100)
            epsilon: Smoothing constant to prevent division by zero (default: 1.0)

        Returns:
            Array of normalized weights (sum to 1)
        """
        # Lower EES (less confident) -> lower weight
        # Higher EES (more confident) -> higher weight
        raw_weights = 1.0 / (epsilon + ees_scores)

        # Normalize to sum to 1
        ensemble_weights = raw_weights / np.sum(raw_weights)

        return ensemble_weights

    def generate_pricing_report(
        self,
        property_ids: np.ndarray,
        q10_cal: np.ndarray,
        q50: np.ndarray,
        q90_cal: np.ndarray,
        uncertainty_score: np.ndarray,
        ees_scores: np.ndarray,
        buy_offers: np.ndarray,
        ensemble_weights: np.ndarray,
    ) -> pd.DataFrame:
        """
        Generate comprehensive pricing and risk report for portfolio.

        Args:
            property_ids: Property identifiers
            q10_cal: Calibrated 10th percentiles
            q50: Median forecasts
            q90_cal: Calibrated 90th percentiles
            uncertainty_score: Uncertainty scores
            ees_scores: Expected Error Scores
            buy_offers: Generated buy-side offers
            ensemble_weights: Ensemble weights for multi-model integration

        Returns:
            DataFrame with full pricing report
        """
        report = pd.DataFrame({
            "property_id": property_ids,
            "median_forecast": q50,
            "q10_lower_bound": q10_cal,
            "q90_upper_bound": q90_cal,
            "interval_width": q90_cal - q10_cal,
            "uncertainty_score": uncertainty_score,
            "ees_score": ees_scores,
            "buy_offer_price": buy_offers,
            "ensemble_weight": ensemble_weights,
            "spread_vs_median": (q50 - buy_offers) / q50 * 100,  # Bid spread %
        })

        # Risk classification
        report["risk_tier"] = pd.cut(
            report["ees_score"],
            bins=[0, 33, 67, 100],
            labels=["High Risk", "Medium Risk", "Low Risk"]
        )

        return report.sort_values("ees_score", ascending=False)


if __name__ == "__main__":
    import os
    from data_loader import RealEstateDataLoader
    from feature_engineering import FeatureEngineer
    from primary_model import PrimaryQuantileAVM
    from meta_error_model import MetaErrorPredictor
    from calibration_engine import CalibrationEngine

    data_path = os.path.join(os.path.dirname(__file__), "kc_house_data.csv")
    loader = RealEstateDataLoader(data_path)
    df = loader.load_data()
    train, val, test = loader.temporal_split()

    # Engineer features
    engineer = FeatureEngineer()
    train_features = engineer.engineer_features(train, fit=True)
    val_features = engineer.engineer_features(val, fit=False)
    test_features = engineer.engineer_features(test, fit=False)

    # Train primary model
    primary = PrimaryQuantileAVM()
    primary.train(train_features, train["price"])

    # Get predictions on test
    q10_raw, q50_raw, q90_raw = primary.predict_with_uncertainty(test_features)

    # Calibrate
    calibrator = CalibrationEngine(target_coverage=0.80)
    calibrator.calibrate(q10_raw, q50_raw, q90_raw, val["price"].values)
    q10_cal, q50_cal, q90_cal = calibrator.apply_calibration(q10_raw, q50_raw, q90_raw)

    # Meta-error predictions
    errors = primary.calculate_raw_errors(train_features, train["price"])
    meta = MetaErrorPredictor()
    meta.train(train_features, errors)
    ees_scores = meta.predict_ees(test_features)

    # Generate offers
    engine = OfferEngine(base_spread=0.05, alpha_uncertainty=0.15)
    uncertainty_score = engine.calculate_uncertainty_score(q10_cal, q50_cal, q90_cal)
    buy_offers = engine.generate_buy_offer(q50_cal, uncertainty_score, ees_scores)
    weights = engine.calculate_ensemble_weights(ees_scores)

    # Report
    report = engine.generate_pricing_report(
        test["id"].values,
        q10_cal, q50_cal, q90_cal,
        uncertainty_score,
        ees_scores,
        buy_offers,
        weights,
    )

    print("\nPricing Report (top 10 highest confidence):")
    print(report.head(10)[["property_id", "median_forecast", "buy_offer_price", "ees_score", "risk_tier"]])
