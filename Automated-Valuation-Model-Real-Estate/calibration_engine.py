"""
Calibration Layer
Learns interval scaling factors to guarantee nominal coverage (e.g., 80%).
"""

import numpy as np
import pandas as pd
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibrationEngine:
    """
    Calibrates quantile prediction intervals to match nominal coverage.
    Learns a scaling factor that stretches [Q_10, Q_90] to achieve target coverage.
    """

    def __init__(self, target_coverage: float = 0.80):
        """
        Initialize Calibration Engine.

        Args:
            target_coverage: Target empirical coverage (default: 0.80 for 80%)
        """
        self.target_coverage = target_coverage
        self.scaling_factor = 1.0
        self.is_calibrated = False

    def calibrate(
        self,
        q10: np.ndarray,
        q50: np.ndarray,
        q90: np.ndarray,
        y_actual: np.ndarray,
        min_scaling: float = 0.9,
        max_scaling: float = 2.0,
    ) -> Tuple[float, float]:
        """
        Learn scaling factor via binary search to achieve target coverage.

        Args:
            q10: Array of 10th percentile predictions
            q50: Array of 50th percentile predictions (median, unchanged)
            q90: Array of 90th percentile predictions
            y_actual: Array of actual prices
            min_scaling: Minimum scaling factor (default: 0.9)
            max_scaling: Maximum scaling factor (default: 2.0)

        Returns:
            Tuple of (scaling_factor, achieved_coverage)
        """
        def coverage_at_scale(scale: float) -> float:
            """Calculate coverage at a given scaling factor"""
            scaled_q10 = q50 - scale * (q50 - q10)
            scaled_q90 = q50 + scale * (q90 - q50)
            
            coverage = np.mean(
                (y_actual >= scaled_q10) & (y_actual <= scaled_q90)
            )
            return coverage

        # Binary search for scaling factor
        low, high = min_scaling, max_scaling
        tolerance = 0.001  # 0.1% tolerance

        for iteration in range(50):
            mid = (low + high) / 2.0
            coverage = coverage_at_scale(mid)

            if coverage < self.target_coverage - tolerance:
                # Need wider intervals
                low = mid
            elif coverage > self.target_coverage + tolerance:
                # Can use narrower intervals
                high = mid
            else:
                # Close enough
                self.scaling_factor = mid
                self.is_calibrated = True
                logger.info(
                    f"Calibration converged in {iteration} iterations. "
                    f"Scaling factor: {self.scaling_factor:.4f}, Coverage: {coverage:.4f}"
                )
                return self.scaling_factor, coverage

        # Fallback if search doesn't converge perfectly
        self.scaling_factor = (low + high) / 2.0
        final_coverage = coverage_at_scale(self.scaling_factor)
        self.is_calibrated = True
        
        logger.warning(
            f"Calibration did not converge fully. "
            f"Scaling factor: {self.scaling_factor:.4f}, Coverage: {final_coverage:.4f}"
        )
        return self.scaling_factor, final_coverage

    def apply_calibration(
        self, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply learned scaling factor to quantile predictions.

        Args:
            q10: Array of 10th percentile predictions
            q50: Array of 50th percentile predictions
            q90: Array of 90th percentile predictions

        Returns:
            Tuple of (q10_calibrated, q50, q90_calibrated)
        """
        if not self.is_calibrated:
            logger.warning("Calibration not fitted yet. Returning uncalibrated predictions.")
            return q10, q50, q90

        q10_cal = q50 - self.scaling_factor * (q50 - q10)
        q90_cal = q50 + self.scaling_factor * (q90 - q50)

        return q10_cal, q50, q90_cal

    def validate_calibration(
        self,
        q10_cal: np.ndarray,
        q50: np.ndarray,
        q90_cal: np.ndarray,
        y_actual: np.ndarray,
    ) -> dict:
        """
        Validate calibration performance on hold-out set.

        Args:
            q10_cal: Calibrated 10th percentile
            q50: 50th percentile (unchanged)
            q90_cal: Calibrated 90th percentile
            y_actual: Actual prices

        Returns:
            Dictionary with validation metrics
        """
        coverage_mask = (y_actual >= q10_cal) & (y_actual <= q90_cal)
        empirical_coverage = coverage_mask.mean()

        interval_widths = q90_cal - q10_cal
        median_width = np.median(interval_widths)
        mean_width = np.mean(interval_widths)

        # Sharpness: tightness of intervals
        sharpness = np.mean(interval_widths / q50)

        metrics = {
            "empirical_coverage": empirical_coverage,
            "target_coverage": self.target_coverage,
            "coverage_error": abs(empirical_coverage - self.target_coverage),
            "median_interval_width": median_width,
            "mean_interval_width": mean_width,
            "sharpness": sharpness,
            "scaling_factor": self.scaling_factor,
        }

        logger.info(
            f"Calibration Validation:\n"
            f"  Empirical Coverage: {empirical_coverage:.4f} (target: {self.target_coverage:.4f})\n"
            f"  Median Interval Width: ${median_width:,.0f}\n"
            f"  Sharpness (interval width / Q_50): {sharpness:.4f}"
        )

        return metrics


if __name__ == "__main__":
    from data_loader import RealEstateDataLoader
    from feature_engineering import FeatureEngineer
    from primary_model import PrimaryQuantileAVM

    # Load and prepare data
    loader = RealEstateDataLoader("/Users/kayleighinman/Downloads/kc_house_data.csv")
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

    # Get predictions on validation set
    q10_raw, q50_raw, q90_raw = primary.predict_with_uncertainty(val_features)

    # Calibrate
    calibrator = CalibrationEngine(target_coverage=0.80)
    calibrator.calibrate(q10_raw, q50_raw, q90_raw, val["price"].values)

    # Apply to test set
    q10_test, q50_test, q90_test = primary.predict_with_uncertainty(test_features)
    q10_cal, q50_cal, q90_cal = calibrator.apply_calibration(q10_test, q50_test, q90_test)

    # Validate
    metrics = calibrator.validate_calibration(q10_cal, q50_cal, q90_cal, test["price"].values)
    print("\nCalibration Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
