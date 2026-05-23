"""
Main Orchestration Script
Integrates data loading, feature engineering, model training, and backtesting
Includes CLI and Streamlit interface
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime, timedelta

# Import custom modules
from data_loader import DataLoader
from features import FeatureEngineer
from model import QuantileModel, BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantileForecastingPipeline:
    """End-to-end quantile return forecasting pipeline"""

    def __init__(
        self,
        alpha_vantage_key: str,
        fred_api_key: str,
        kalshi_api_key: Optional[str] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            alpha_vantage_key: Alpha Vantage API key
            fred_api_key: FRED API key
            kalshi_api_key: Optional Kalshi API key
        """
        self.data_loader = DataLoader(
            alpha_vantage_key=alpha_vantage_key,
            fred_api_key=fred_api_key,
            kalshi_api_key=kalshi_api_key,
        )
        self.feature_engineer = FeatureEngineer(lookback_days=252)
        self.model = None
        self.backtest_results = None

    def load_and_prepare_data(
        self, ticker: str, min_market_cap: float = 2e10
    ) -> Optional[Dict]:
        """
        Load price, fundamental, and macro data for a ticker.

        Args:
            ticker: Stock ticker symbol
            min_market_cap: Minimum market cap filter ($20B default)

        Returns:
            Dictionary with price data, features, and metadata, or None if filtered out
        """
        logger.info(f"Loading data for {ticker}...")

        # Check market cap
        fundamentals = self.data_loader.fetch_alpha_vantage_fundamentals(ticker)
        market_cap = fundamentals.get("market_cap", 0)

        if market_cap < min_market_cap:
            logger.warning(
                f"{ticker} market cap (${market_cap/1e9:.2f}B) below threshold of ${min_market_cap/1e9:.2f}B"
            )
            return None

        logger.info(f"{ticker} qualifies (${market_cap/1e9:.2f}B market cap)")

        # Fetch price data
        price_data = self.data_loader.fetch_alpha_vantage_daily(ticker)
        if price_data.empty:
            logger.error(f"No price data retrieved for {ticker}")
            return None

        # Fetch macro data
        macro_data = {}
        macro_series = {
            "DGS10": "10-Year Yield",
            "CPIAUCSL": "CPI",
            "A191RL1Q225SBEA": "Real GDP Growth",
        }

        start_date = price_data.index[0].strftime("%Y-%m-%d")
        for series_id, label in macro_series.items():
            df = self.data_loader.fetch_fred_series(series_id, start_date)
            if not df.empty:
                macro_data[series_id] = df.iloc[:, 0]

        macro_df = pd.DataFrame(macro_data) if macro_data else pd.DataFrame()

        return {
            "ticker": ticker,
            "price_data": price_data,
            "macro_data": macro_df,
            "fundamentals": fundamentals,
        }

    def engineer_features(self, data_dict: Dict) -> Optional[pd.DataFrame]:
        """
        Generate feature set from raw data.

        Args:
            data_dict: Output from load_and_prepare_data

        Returns:
            Feature DataFrame or None if insufficient data
        """
        if not data_dict:
            return None

        logger.info(f"Engineering features for {data_dict['ticker']}...")

        features = self.feature_engineer.engineer_features(
            data_dict["price_data"],
            data_dict["macro_data"],
            kalshi_features=None,  # Can add Kalshi features here
        )

        if features.empty or len(features) < 500:
            logger.error(f"Insufficient features after engineering (n={len(features)})")
            return None

        logger.info(f"Generated {len(features)} feature vectors with {len(features.columns)} dimensions")
        return features

    def train_model(
        self,
        features: pd.DataFrame,
        quantiles: Optional[list] = None,
    ) -> Optional[QuantileModel]:
        """
        Train quantile regression model.

        Args:
            features: Feature DataFrame
            quantiles: Quantiles to predict (default: [0.2, 0.5, 0.8])

        Returns:
            Trained QuantileModel or None
        """
        if features is None or features.empty:
            logger.error("No features available for training")
            return None

        logger.info("Training quantile regression models...")

        quantiles = quantiles or [0.2, 0.5, 0.8]
        model = QuantileModel(quantiles=quantiles)

        # Separate features from target
        y = features["forward_return_pct"]
        X = features.drop(columns=["forward_return_pct"])

        # Train-test split
        split_idx = int(0.8 * len(X))
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        model.train(X_train, y_train, verbose=0)
        self.model = model

        logger.info("Model training complete")
        return model

    def run_backtest(
        self,
        features: pd.DataFrame,
        prices: pd.Series,
        train_window: int = 504,
        test_window: int = 63,
    ) -> pd.DataFrame:
        """
        Execute walk-forward backtest.

        Args:
            features: Feature DataFrame
            prices: Price series
            train_window: Training window (trading days)
            test_window: Test window (trading days)

        Returns:
            DataFrame with backtest results
        """
        if self.model is None:
            logger.error("Model not trained. Run train_model first.")
            return pd.DataFrame()

        logger.info("Running walk-forward backtest...")

        y = features["forward_return_pct"]
        X = features.drop(columns=["forward_return_pct"])

        engine = BacktestEngine(initial_price=prices.iloc[0])
        results = engine.run_walk_forward_backtest(
            X, prices, y, self.model,
            train_window=train_window,
            test_window=test_window,
        )

        self.backtest_results = results
        logger.info(f"Backtest complete: {len(results)} results generated")
        logger.info(engine.generate_backtest_report(results))

        return results

    def forecast_prices(
        self, ticker: str, quantiles: Optional[list] = None
    ) -> Optional[Dict]:
        """
        Generate forward price forecasts for a ticker.

        Args:
            ticker: Stock ticker symbol
            quantiles: Quantiles to forecast (default: [0.2, 0.5, 0.8])

        Returns:
            Dictionary with forecasted prices at different quantiles
        """
        quantiles = quantiles or [0.2, 0.5, 0.8]

        # Load data
        data_dict = self.load_and_prepare_data(ticker)
        if not data_dict:
            return None

        # Engineer features
        features = self.engineer_features(data_dict)
        if features is None:
            return None

        # Train model
        model = self.train_model(features, quantiles=quantiles)
        if model is None:
            return None

        # Use most recent data for forecast
        y = features["forward_return_pct"]
        X = features.drop(columns=["forward_return_pct"])
        current_price = data_dict["price_data"]["Close"].iloc[-1]

        # Retrain on all data for final forecast
        model.train(X, y, verbose=0)

        # Predict for current date
        X_latest = X.iloc[-1:].values.reshape(1, -1)
        predicted_returns = model.predict(X_latest)

        # Reconstruct prices
        forecast_prices = {
            "ticker": ticker,
            "current_price": float(current_price),
            "forecast_date": (datetime.now() + timedelta(days=63)).strftime("%Y-%m-%d"),
            "predictions": {}
        }

        for quantile, returns in predicted_returns.items():
            forecasted_price = current_price * (1 + returns[0] / 100.0)
            forecast_prices["predictions"][f"quantile_{quantile}"] = {
                "return_pct": float(returns[0]),
                "forecasted_price": float(forecasted_price),
            }

        return forecast_prices


def cli_main():
    """Command-line interface for the pipeline"""

    parser = argparse.ArgumentParser(
        description="Quantile Return Forecasting Pipeline"
    )
    parser.add_argument(
        "--alpha-vantage-key",
        required=True,
        help="Alpha Vantage API key",
    )
    parser.add_argument(
        "--fred-key",
        required=True,
        help="FRED API key",
    )
    parser.add_argument(
        "--kalshi-key",
        default=None,
        help="Kalshi API key (optional)",
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Stock ticker to forecast",
    )
    parser.add_argument(
        "--mode",
        choices=["forecast", "backtest"],
        default="forecast",
        help="Run mode: forecast or backtest",
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = QuantileForecastingPipeline(
        alpha_vantage_key=args.alpha_vantage_key,
        fred_api_key=args.fred_key,
        kalshi_api_key=args.kalshi_key,
    )

    if args.mode == "forecast":
        logger.info(f"Generating forecast for {args.ticker}...")
        forecast = pipeline.forecast_prices(args.ticker)
        if forecast:
            logger.info("\n=== PRICE FORECAST ===")
            logger.info(f"Ticker: {forecast['ticker']}")
            logger.info(f"Current Price: ${forecast['current_price']:.2f}")
            logger.info(f"Forecast Date: {forecast['forecast_date']}")
            for key, pred in forecast["predictions"].items():
                logger.info(
                    f"{key}: ${pred['forecasted_price']:.2f} "
                    f"({pred['return_pct']:+.2f}%)"
                )

    elif args.mode == "backtest":
        logger.info(f"Running backtest for {args.ticker}...")
        data_dict = pipeline.load_and_prepare_data(args.ticker)
        if data_dict:
            features = pipeline.engineer_features(data_dict)
            if features is not None:
                pipeline.train_model(features)
                pipeline.run_backtest(
                    features,
                    data_dict["price_data"]["Close"],
                )


def streamlit_interface():
    """Streamlit web interface for the pipeline"""
    try:
        import streamlit as st
    except ImportError:
        logger.error("Streamlit not installed. Install with: pip install streamlit")
        return

    st.set_page_config(page_title="Quantile Return Forecaster", layout="wide")

    st.title("📈 Quantile Return Forecasting System")
    st.markdown(
        """
    This app predicts 3-month forward stock prices using quantile regression,
    displaying a "fan chart" of predicted prices at the 20th, 50th, and 80th percentiles.
    """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        alpha_key = st.text_input("Alpha Vantage API Key", type="password")
    with col2:
        fred_key = st.text_input("FRED API Key", type="password")
    with col3:
        kalshi_key = st.text_input("Kalshi API Key (optional)", type="password")

    ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()

    if st.button("Generate Forecast", key="forecast_btn"):
        if not alpha_key or not fred_key:
            st.error("Please provide Alpha Vantage and FRED API keys")
        else:
            with st.spinner("Loading data and training model..."):
                pipeline = QuantileForecastingPipeline(
                    alpha_vantage_key=alpha_key,
                    fred_api_key=fred_key,
                    kalshi_api_key=kalshi_key,
                )

                forecast = pipeline.forecast_prices(ticker)

                if forecast:
                    st.success("✅ Forecast Generated!")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Current Price",
                            f"${forecast['current_price']:.2f}",
                        )
                    with col2:
                        st.metric(
                            "Forecast Date",
                            forecast['forecast_date'],
                        )

                    st.subheader("Price Forecasts (3-Month Forward)")

                    forecast_data = []
                    for key, pred in forecast["predictions"].items():
                        quantile = key.replace("quantile_", "")
                        forecast_data.append({
                            "Percentile": f"{float(quantile)*100:.0f}th",
                            "Forecasted Price": f"${pred['forecasted_price']:.2f}",
                            "Return": f"{pred['return_pct']:+.2f}%",
                        })

                    st.dataframe(forecast_data, use_container_width=True)

                    # Fan chart visualization
                    prices = [
                        forecast["predictions"]["quantile_0.2"]["forecasted_price"],
                        forecast["predictions"]["quantile_0.5"]["forecasted_price"],
                        forecast["predictions"]["quantile_0.8"]["forecasted_price"],
                    ]

                    import plotly.graph_objects as go

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            y=[forecast["current_price"]] + prices,
                            x=["Current", "20th %ile", "Median", "80th %ile"],
                            mode="lines+markers",
                            name="Price Forecast",
                            line=dict(color="rgba(0, 100, 200, 0.8)"),
                        )
                    )

                    fig.update_layout(
                        title="Price Forecast Fan Chart",
                        xaxis_title="Quantile",
                        yaxis_title="Price ($)",
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(f"Could not generate forecast for {ticker}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        streamlit_interface()
    else:
        cli_main()
