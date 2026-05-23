"""
Feature Engineering & Preprocessing Module
Handles technical indicators, macro features, and stationarity checks
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generates and preprocesses features for quantile return forecasting"""

    TRADING_DAYS = 252
    FORECAST_HORIZON = 63  # 3 months

    def __init__(self, lookback_days: int = 252):
        """
        Initialize FeatureEngineer.

        Args:
            lookback_days: Historical period for calculating indicators
        """
        self.lookback_days = lookback_days

    @staticmethod
    def calculate_forward_returns(
        prices: pd.Series, horizon: int = 63
    ) -> pd.Series:
        """
        Calculate forward-looking returns at specified horizon.

        Args:
            prices: Series of prices (typically Adj Close)
            horizon: Trading days forward (default: 63 for 3 months)

        Returns:
            Series of forward returns as percentages
        """
        # Shift prices backward to align with current date
        future_prices = prices.shift(-horizon)
        forward_returns = (future_prices - prices) / prices * 100

        return forward_returns

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            prices: Series of prices
            period: RSI period (default: 14)

        Returns:
            Series of RSI values (0-100)
        """
        deltas = prices.diff()
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100.0 - 100.0 / (1.0 + rs)

        for i in range(period, len(prices)):
            delta = deltas[i]
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period

            rs = up / down if down != 0 else 0
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

        return pd.Series(rsi, index=prices.index)

    @staticmethod
    def calculate_macd(
        prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD and Signal line.

        Args:
            prices: Series of prices
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)

        Returns:
            Tuple of (MACD, Signal, Histogram)
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal

        return macd, macd_signal, macd_hist

    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        """
        Calculate log returns for stationarity.

        Args:
            prices: Series of prices

        Returns:
            Series of log returns
        """
        return np.log(prices / prices.shift(1)) * 100

    @staticmethod
    def calculate_momentum(prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate price momentum.

        Args:
            prices: Series of prices
            period: Lookback period (default: 20 days)

        Returns:
            Series of momentum values
        """
        return (prices - prices.shift(period)) / prices.shift(period) * 100

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series, period: int = 20, std_dev: int = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Series of prices
            period: MA period (default: 20)
            std_dev: Number of standard deviations (default: 2)

        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)

        return upper_band, sma, lower_band

    @staticmethod
    def check_stationarity(series: pd.Series, name: str = "") -> bool:
        """
        Check if a series is stationary using Augmented Dickey-Fuller test.

        Args:
            series: Series to test
            name: Name for logging

        Returns:
            True if stationary (p-value < 0.05), False otherwise
        """
        try:
            from statsmodels.tsa.stattools import adfuller

            result = adfuller(series.dropna(), autolag="AIC")
            p_value = result[1]

            is_stationary = p_value < 0.05
            logger.info(
                f"{name} - ADF p-value: {p_value:.4f} - {'Stationary' if is_stationary else 'Non-Stationary'}"
            )
            return is_stationary
        except ImportError:
            logger.warning("statsmodels not installed. Skipping stationarity test.")
            return True

    def engineer_features(
        self, ohlcv_data: pd.DataFrame, macro_data: pd.DataFrame,
        kalshi_features: Optional[Dict[str, pd.Series]] = None,
    ) -> pd.DataFrame:
        """
        Create comprehensive feature set for modeling.

        Args:
            ohlcv_data: DataFrame with OHLCV data
            macro_data: DataFrame with macro features
            kalshi_features: Optional dict of sentiment features

        Returns:
            DataFrame with all engineered features
        """
        features = ohlcv_data[["Close", "Volume"]].copy()

        # Technical Indicators
        features["rsi_14"] = self.calculate_rsi(ohlcv_data["Close"], period=14)
        macd, macd_signal, macd_hist = self.calculate_macd(ohlcv_data["Close"])
        features["macd"] = macd
        features["macd_signal"] = macd_signal
        features["macd_hist"] = macd_hist

        upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(
            ohlcv_data["Close"]
        )
        features["bb_upper"] = upper_bb
        features["bb_middle"] = middle_bb
        features["bb_lower"] = lower_bb

        features["momentum_20"] = self.calculate_momentum(ohlcv_data["Close"], 20)
        features["log_returns"] = self.calculate_log_returns(ohlcv_data["Close"])

        # Volume features
        features["volume_sma_20"] = ohlcv_data["Volume"].rolling(20).mean()
        features["volume_ratio"] = (
            ohlcv_data["Volume"] / features["volume_sma_20"]
        )

        # Price-based features
        features["price_sma_50"] = ohlcv_data["Close"].rolling(50).mean()
        features["price_sma_200"] = ohlcv_data["Close"].rolling(200).mean()
        features["sma_ratio"] = features["price_sma_50"] / features["price_sma_200"]

        # Merge macro features
        if not macro_data.empty:
            features = features.join(macro_data, how="left")

        # Add Kalshi sentiment features
        if kalshi_features:
            for feature_name, feature_series in kalshi_features.items():
                features[f"kalshi_{feature_name}"] = feature_series

        # Calculate target (forward returns) - should be done separately during preprocessing
        features["forward_return_pct"] = self.calculate_forward_returns(
            ohlcv_data["Close"], horizon=self.FORECAST_HORIZON
        )

        # Drop rows with NaN values
        features = features.dropna()

        return features

    @staticmethod
    def prepare_train_test_split(
        features: pd.DataFrame,
        target: pd.Series,
        test_size: float = 0.2,
        walk_forward: bool = True,
    ) -> Tuple[Tuple, Tuple]:
        """
        Create train-test split with walk-forward validation option.

        Args:
            features: Feature DataFrame
            target: Target series (forward returns)
            test_size: Proportion for test set (0-1)
            walk_forward: If True, use temporal split; if False, random split

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if walk_forward:
            split_idx = int(len(features) * (1 - test_size))
            X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
        else:
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=test_size, random_state=42
            )

        return (X_train, X_test, y_train, y_test)

    @staticmethod
    def normalize_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame
    ]:
        """
        Standardize features using mean/std from training set.

        Args:
            X_train: Training feature set
            X_test: Test feature set

        Returns:
            Tuple of (normalized X_train, normalized X_test)
        """
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(
            X_train_scaled, index=X_train.index, columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled, index=X_test.index, columns=X_test.columns
        )

        return X_train_scaled, X_test_scaled


if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()

    # Simulate OHLCV data
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    prices = 100 + np.cumsum(np.random.randn(1000) * 2)
    ohlcv = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + np.abs(np.random.randn(1000)),
            "Low": prices - np.abs(np.random.randn(1000)),
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, 1000),
        },
        index=dates,
    )

    # Simulate macro data
    macro = pd.DataFrame(
        {
            "DGS10": 2.5 + np.cumsum(np.random.randn(1000) * 0.01),
            "CPIAUCSL": 300 + np.cumsum(np.random.randn(1000) * 0.1),
        },
        index=dates,
    )

    features = engineer.engineer_features(ohlcv, macro)
    print("Features head:")
    print(features.head())
    print("\nFeatures shape:", features.shape)
