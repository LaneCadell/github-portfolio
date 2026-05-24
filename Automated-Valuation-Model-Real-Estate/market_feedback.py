"""
Market Feedback & Dynamic Adjustment Layer
Tracks sell-side resale performance for post-hoc macro bias mitigation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketFeedbackLoop:
    """
    Monitors realized resale prices against model expectations.
    Generates rolling Market Multiplier to adjust forecasts for macro shifts.
    """

    def __init__(self, lookback_window: int = 252):
        """
        Initialize Market Feedback Loop.

        Args:
            lookback_window: Number of recent transactions for rolling median (default: 252 ~ 1 year)
        """
        self.lookback_window = lookback_window
        self.transaction_log = []
        self.market_multiplier = 1.0

    def record_transaction(
        self,
        property_id: int,
        q50_forecast: float,
        actual_resale_price: float,
        date: str,
        bought_at_price: Optional[float] = None,
    ) -> None:
        """
        Record a completed resale transaction.

        Args:
            property_id: Unique property identifier
            q50_forecast: Original median forecast used for pricing decision
            actual_resale_price: Realized resale price
            date: Transaction date
            bought_at_price: Purchase price (optional, for performance tracking)
        """
        sale_ratio = actual_resale_price / (q50_forecast + 1e-10)

        transaction = {
            "property_id": property_id,
            "q50_forecast": q50_forecast,
            "actual_resale_price": actual_resale_price,
            "sale_ratio": sale_ratio,
            "date": date,
            "bought_at_price": bought_at_price,
        }

        self.transaction_log.append(transaction)

        logger.debug(
            f"Transaction recorded: prop_id={property_id}, "
            f"q50={q50_forecast:,.0f}, resale={actual_resale_price:,.0f}, "
            f"ratio={sale_ratio:.4f}"
        )

    def update_market_multiplier(self) -> float:
        """
        Update market multiplier based on rolling median of recent sale ratios.
        
        $$\text{Market Multiplier} = \text{median}(\text{recent Sale Ratios})$$

        Returns:
            Updated market multiplier
        """
        if len(self.transaction_log) == 0:
            logger.warning("No transactions recorded. Market multiplier remains 1.0")
            return self.market_multiplier

        # Use recent transactions
        recent_txns = self.transaction_log[-self.lookback_window:]
        sale_ratios = [txn["sale_ratio"] for txn in recent_txns]

        self.market_multiplier = np.median(sale_ratios)

        logger.info(
            f"Updated Market Multiplier to {self.market_multiplier:.4f} "
            f"(based on {len(recent_txns)} recent transactions)"
        )

        return self.market_multiplier

    def apply_dynamic_shift(
        self,
        q10_cal: np.ndarray,
        q50: np.ndarray,
        q90_cal: np.ndarray,
    ) -> tuple:
        """
        Apply post-hoc market multiplier adjustment to mitigate macro bias.
        
        $$[Q_{10\_fb}, Q_{50\_fb}, Q_{90\_fb}] = [Q_{10\_cal}, Q_{50}, Q_{90\_cal}] \times \text{Market Multiplier}$$

        Args:
            q10_cal: Calibrated 10th percentiles
            q50: 50th percentiles (medians)
            q90_cal: Calibrated 90th percentiles

        Returns:
            Tuple of (q10_feedback, q50_feedback, q90_feedback)
        """
        q10_feedback = q10_cal * self.market_multiplier
        q50_feedback = q50 * self.market_multiplier
        q90_feedback = q90_cal * self.market_multiplier

        logger.debug(
            f"Applied market multiplier {self.market_multiplier:.4f} to forecasts"
        )

        return q10_feedback, q50_feedback, q90_feedback

    def generate_performance_report(self) -> pd.DataFrame:
        """
        Generate summary of realized transaction performance.

        Returns:
            DataFrame with transaction history and metrics
        """
        if not self.transaction_log:
            return pd.DataFrame()

        df_txns = pd.DataFrame(self.transaction_log)

        report = {
            "total_transactions": len(df_txns),
            "median_sale_ratio": df_txns["sale_ratio"].median(),
            "mean_sale_ratio": df_txns["sale_ratio"].mean(),
            "sale_ratio_std": df_txns["sale_ratio"].std(),
            "median_forecast_error": (
                (df_txns["actual_resale_price"] - df_txns["q50_forecast"]) /
                df_txns["q50_forecast"]
            ).median(),
            "portfolio_pnl_vs_forecast": (
                (df_txns["actual_resale_price"] - df_txns["q50_forecast"]).sum()
            ),
        }

        if "bought_at_price" in df_txns.columns and df_txns["bought_at_price"].notna().any():
            bought_mask = df_txns["bought_at_price"].notna()
            report["purchase_to_resale_gain"] = (
                (df_txns.loc[bought_mask, "actual_resale_price"] -
                 df_txns.loc[bought_mask, "bought_at_price"]) /
                df_txns.loc[bought_mask, "bought_at_price"]
            ).median()

        logger.info("Performance Report:")
        for key, value in report.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

        return pd.DataFrame([report])

    def get_transaction_log(self) -> pd.DataFrame:
        """
        Export transaction log as DataFrame.

        Returns:
            DataFrame of all recorded transactions
        """
        if not self.transaction_log:
            return pd.DataFrame()

        return pd.DataFrame(self.transaction_log)


if __name__ == "__main__":
    # Example usage
    feedback_loop = MarketFeedbackLoop(lookback_window=10)

    # Simulate transactions
    np.random.seed(42)
    for i in range(20):
        q50_forecast = 500000 + np.random.randn() * 100000
        # Market moved up 3% on average
        actual_resale = q50_forecast * (1.03 + np.random.randn() * 0.05)
        feedback_loop.record_transaction(
            property_id=1000 + i,
            q50_forecast=q50_forecast,
            actual_resale_price=actual_resale,
            date=f"2025-{(i % 12) + 1:02d}-01",
            bought_at_price=q50_forecast * 0.95,
        )

    # Update multiplier
    feedback_loop.update_market_multiplier()

    # Apply to example forecasts
    q10_cal = np.array([450000, 500000, 550000])
    q50 = np.array([500000, 550000, 600000])
    q90_cal = np.array([550000, 600000, 650000])

    q10_adj, q50_adj, q90_adj = feedback_loop.apply_dynamic_shift(q10_cal, q50, q90_cal)

    print("Original forecasts:")
    print(f"  Q50: {q50}")
    print("\nAfter market multiplier adjustment:")
    print(f"  Q50 (adjusted): {q50_adj}")
    print(f"  Multiplier: {feedback_loop.market_multiplier:.4f}")

    # Performance report
    print("\n" + feedback_loop.generate_performance_report().to_string())
