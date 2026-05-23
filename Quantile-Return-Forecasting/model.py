"""
Model Training & Backtesting Module
LightGBM quantile regression with walk-forward validation
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import logging
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import spearmanr
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantileModel:
    """LightGBM-based quantile regression model for return forecasting"""

    def __init__(
        self,
        quantiles: List[float] = None,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 7,
        seed: int = 42,
    ):
        """
        Initialize QuantileModel.

        Args:
            quantiles: List of quantiles to predict (default: [0.2, 0.5, 0.8])
            n_estimators: Number of boosting rounds
            learning_rate: LightGBM learning rate
            max_depth: Max tree depth
            seed: Random seed
        """
        self.quantiles = quantiles or [0.2, 0.5, 0.8]
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.seed = seed
        self.models = {}  # Dict to store one model per quantile
        self.feature_names = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        verbose: int = 0,
    ) -> None:
        """
        Train quantile regression models.

        Args:
            X_train: Training features
            y_train: Training targets (forward returns in %)
            verbose: LightGBM verbosity level
        """
        self.feature_names = X_train.columns.tolist()

        for quantile in self.quantiles:
            logger.info(f"Training model for quantile {quantile}...")

            params = {
                "objective": "quantile",
                "alpha": quantile,
                "metric": "quantile",
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "num_leaves": 2 ** self.max_depth,
                "verbose": verbose,
                "seed": self.seed,
                "min_data_in_leaf": 20,
            }

            train_data = lgb.Dataset(X_train, label=y_train)

            self.models[quantile] = lgb.train(
                params,
                train_data,
                num_boost_round=self.n_estimators,
                verbose_eval=False,
            )

        logger.info(f"Successfully trained models for {len(self.models)} quantiles")

    def predict(self, X_test: pd.DataFrame) -> Dict[float, np.ndarray]:
        """
        Generate predictions for all quantiles.

        Args:
            X_test: Test features

        Returns:
            Dictionary mapping quantile -> predictions
        """
        predictions = {}
        for quantile, model in self.models.items():
            predictions[quantile] = model.predict(X_test)

        return predictions

    def get_feature_importance(self, quantile: float = 0.5, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance for a specific quantile model.

        Args:
            quantile: Quantile for which to get importance (default: 0.5)
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if quantile not in self.models:
            logger.warning(f"Model for quantile {quantile} not found")
            return pd.DataFrame()

        model = self.models[quantile]
        importance = model.feature_importance()

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)

        return importance_df.head(top_n)


class BacktestEngine:
    """Walk-forward backtesting engine with comprehensive metrics"""

    def __init__(self, initial_price: float, forecast_horizon: int = 63):
        """
        Initialize BacktestEngine.

        Args:
            initial_price: Starting price for reconstruction
            forecast_horizon: Trading days forward (default: 63)
        """
        self.initial_price = initial_price
        self.forecast_horizon = forecast_horizon
        self.results = []

    @staticmethod
    def reconstruct_prices(
        current_prices: pd.Series,
        predicted_returns: Dict[float, np.ndarray],
    ) -> Dict[float, np.ndarray]:
        """
        Convert predicted returns to price forecasts.

        Args:
            current_prices: Current prices at prediction time
            predicted_returns: Dict mapping quantile -> return %

        Returns:
            Dict mapping quantile -> predicted prices
        """
        predicted_prices = {}
        for quantile, returns_pct in predicted_returns.items():
            predicted_prices[quantile] = current_prices.values * (1 + returns_pct / 100.0)

        return predicted_prices

    @staticmethod
    def calculate_directional_accuracy(
        predicted_returns: np.ndarray, actual_returns: np.ndarray
    ) -> float:
        """
        Calculate directional accuracy (% correct sign predictions).

        Args:
            predicted_returns: Predicted returns
            actual_returns: Realized returns

        Returns:
            Directional accuracy (0-1)
        """
        predicted_sign = np.sign(predicted_returns)
        actual_sign = np.sign(actual_returns)
        correct = np.sum(predicted_sign == actual_sign)
        da = correct / len(actual_returns)
        return da

    @staticmethod
    def calculate_information_coefficient(
        predicted_returns: np.ndarray, actual_returns: np.ndarray
    ) -> float:
        """
        Calculate Information Coefficient (Spearman rank correlation).

        Args:
            predicted_returns: Predicted returns
            actual_returns: Realized returns

        Returns:
            Spearman correlation coefficient
        """
        ic, p_value = spearmanr(predicted_returns, actual_returns)
        return ic if not np.isnan(ic) else 0.0

    def run_walk_forward_backtest(
        self,
        features: pd.DataFrame,
        prices: pd.Series,
        target: pd.Series,
        model: QuantileModel,
        train_window: int = 504,  # 2 years of trading days
        test_window: int = 63,   # 3 months
    ) -> pd.DataFrame:
        """
        Execute walk-forward backtest.

        Args:
            features: Feature DataFrame
            prices: Price series
            target: Target forward returns
            model: Trained QuantileModel
            train_window: Training window in days
            test_window: Test window in days

        Returns:
            DataFrame with backtest results
        """
        results = []
        i = train_window

        while i + test_window <= len(features):
            # Training period
            train_idx = slice(i - train_window, i)
            X_train = features.iloc[train_idx]
            y_train = target.iloc[train_idx]

            # Test period
            test_idx = slice(i, i + test_window)
            X_test = features.iloc[test_idx]
            y_test = target.iloc[test_idx]
            test_prices = prices.iloc[test_idx]

            # Train model
            model.train(X_train, y_train, verbose=0)

            # Predict
            predicted_returns = model.predict(X_test)
            predicted_prices = self.reconstruct_prices(
                test_prices, predicted_returns
            )

            # Evaluate
            for quantile in model.quantiles:
                pred_ret = predicted_returns[quantile]
                da = self.calculate_directional_accuracy(pred_ret, y_test.values)
                ic = self.calculate_information_coefficient(pred_ret, y_test.values)
                mae = mean_absolute_error(y_test.values, pred_ret)
                mape = mean_absolute_percentage_error(y_test.values, pred_ret)

                results.append(
                    {
                        "fold": len(results) // len(model.quantiles),
                        "quantile": quantile,
                        "date": features.index[i],
                        "directional_accuracy": da,
                        "information_coefficient": ic,
                        "mae_returns": mae,
                        "mape_returns": mape,
                        "mae_prices": mean_absolute_error(
                            test_prices.values, predicted_prices[quantile]
                        ),
                        "mape_prices": mean_absolute_percentage_error(
                            test_prices.values, predicted_prices[quantile]
                        ),
                    }
                )

            i += test_window
            logger.info(f"Completed fold {len(results) // len(model.quantiles)}")

        results_df = pd.DataFrame(results)
        return results_df

    @staticmethod
    def generate_backtest_report(results_df: pd.DataFrame) -> str:
        """
        Generate summary statistics from backtest results.

        Args:
            results_df: DataFrame from run_walk_forward_backtest

        Returns:
            Formatted report string
        """
        report = "\n=== BACKTEST RESULTS SUMMARY ===\n"

        for quantile in results_df["quantile"].unique():
            quant_results = results_df[results_df["quantile"] == quantile]

            report += f"\n--- Quantile {quantile} ---\n"
            report += f"Directional Accuracy: {quant_results['directional_accuracy'].mean():.4f} "
            report += f"(std: {quant_results['directional_accuracy'].std():.4f})\n"
            report += f"Information Coefficient: {quant_results['information_coefficient'].mean():.4f} "
            report += f"(std: {quant_results['information_coefficient'].std():.4f})\n"
            report += f"MAE (Returns %): {quant_results['mae_returns'].mean():.4f}\n"
            report += f"MAPE (Returns %): {quant_results['mape_returns'].mean():.4f}\n"
            report += f"MAE (Prices): ${quant_results['mae_prices'].mean():.2f}\n"
            report += f"MAPE (Prices): {quant_results['mape_prices'].mean():.4f}\n"

        report += "\n=== OVERALL STATISTICS ===\n"
        report += f"Average Directional Accuracy: {results_df['directional_accuracy'].mean():.4f}\n"
        report += f"Average Information Coefficient: {results_df['information_coefficient'].mean():.4f}\n"

        return report


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Simulate data
    n_samples = 1000
    n_features = 20
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = 5 + 2 * X.iloc[:, 0] - 1.5 * X.iloc[:, 1] + np.random.randn(n_samples) * 3
    prices = pd.Series(100 + np.cumsum(np.random.randn(n_samples) * 0.5))

    # Create and train model
    model = QuantileModel(quantiles=[0.2, 0.5, 0.8])

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train
    model.train(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)
    print("Predictions (0.5 quantile):", predictions[0.5][:5])

    # Backtest
    engine = BacktestEngine(initial_price=100.0)
    print("BacktestEngine initialized and ready for walk-forward validation")
