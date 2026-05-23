"""
Data Acquisition Module
Handles API integrations for Alpha Vantage, FRED, and Kalshi
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Centralized data acquisition from multiple financial APIs"""

    def __init__(
        self,
        alpha_vantage_key: str,
        fred_api_key: str,
        kalshi_api_key: Optional[str] = None,
        rate_limit_delay: float = 0.5,
    ):
        """
        Initialize DataLoader with API credentials.

        Args:
            alpha_vantage_key: Alpha Vantage API key
            fred_api_key: FRED API key
            kalshi_api_key: Kalshi API key (optional)
            rate_limit_delay: Delay between API calls in seconds
        """
        self.alpha_vantage_key = alpha_vantage_key
        self.fred_api_key = fred_api_key
        self.kalshi_api_key = kalshi_api_key
        self.rate_limit_delay = rate_limit_delay

    def _rate_limit_delay(self):
        """Enforce rate limiting between API calls."""
        time.sleep(self.rate_limit_delay)

    def fetch_alpha_vantage_daily(
        self, ticker: str, outputsize: str = "full"
    ) -> pd.DataFrame:
        """
        Fetch daily adjusted OHLCV data from Alpha Vantage.

        Args:
            ticker: Stock ticker symbol
            outputsize: 'compact' or 'full'

        Returns:
            DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
        """
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": outputsize,
            "apikey": self.alpha_vantage_key,
        }

        try:
            self._rate_limit_delay()
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data:
                logger.error(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
                return pd.DataFrame()

            if "Time Series (Daily)" not in data:
                logger.warning(
                    f"No time series data for {ticker}. API might be rate-limited."
                )
                return pd.DataFrame()

            df = pd.DataFrame.from_dict(
                data["Time Series (Daily)"], orient="index"
            ).astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            df.columns = [
                "Open",
                "High",
                "Low",
                "Close",
                "AdjClose",
                "Volume",
                "DividendAmount",
                "SplitCoefficient",
            ]

            return df[["Open", "High", "Low", "Close", "AdjClose", "Volume"]]

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_alpha_vantage_fundamentals(self, ticker: str) -> Dict:
        """
        Fetch company fundamentals including market cap from Alpha Vantage.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with market cap and other company info
        """
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "OVERVIEW",
            "symbol": ticker,
            "apikey": self.alpha_vantage_key,
        }

        try:
            self._rate_limit_delay()
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data:
                logger.error(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
                return {}

            market_cap = data.get("MarketCapitalization", "0")
            try:
                market_cap = float(market_cap) if market_cap else 0
            except (ValueError, TypeError):
                market_cap = 0

            return {
                "symbol": ticker,
                "market_cap": market_cap,
                "description": data.get("Description", ""),
                "pe_ratio": float(data.get("PERatio", 0) or 0),
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Fundamentals API request failed for {ticker}: {e}")
            return {}

    def fetch_fred_series(self, series_id: str, start_date: str = None) -> pd.DataFrame:
        """
        Fetch macroeconomic data from FRED.

        Args:
            series_id: FRED series identifier (e.g., 'DGS10', 'CPIAUCSL', 'A191RL1Q225SBEA')
            start_date: Start date for data (YYYY-MM-DD format)

        Returns:
            DataFrame with date index and series values
        """
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
        }

        if start_date:
            params["observation_start"] = start_date

        try:
            self._rate_limit_delay()
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "observations" not in data:
                logger.warning(f"No observations for FRED series {series_id}")
                return pd.DataFrame()

            observations = data["observations"]
            dates = []
            values = []

            for obs in observations:
                try:
                    date = pd.to_datetime(obs["date"])
                    value = float(obs["value"])
                    dates.append(date)
                    values.append(value)
                except (ValueError, KeyError):
                    continue

            df = pd.DataFrame({"date": dates, series_id: values})
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed for {series_id}: {e}")
            return pd.DataFrame()

    def fetch_kalshi_odds(
        self, event_ticker: str, date: str = None
    ) -> Optional[float]:
        """
        Fetch market-based probability data from Kalshi for a specific event.

        Args:
            event_ticker: Kalshi event ticker (e.g., 'RCUT25_DEC')
            date: Date for historical odds lookup (YYYY-MM-DD)

        Returns:
            Probability between 0 and 1, or None if unavailable
        """
        if not self.kalshi_api_key:
            logger.warning("Kalshi API key not provided")
            return None

        url = f"https://api.kalshi.com/trade-api/v2/events/{event_ticker}"

        try:
            self._rate_limit_delay()
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "event" in data:
                event_data = data["event"]
                # Extract implied probability from yes contract price
                # Kalshi yes contract price ~ probability
                last_yes_price = event_data.get("last_price_yes", None)
                if last_yes_price is not None:
                    return float(last_yes_price) / 100.0

            return None

        except requests.exceptions.RequestException as e:
            logger.warning(f"Kalshi API request failed for {event_ticker}: {e}")
            return None

    def filter_by_market_cap(
        self, tickers: List[str], min_market_cap: float = 2e10
    ) -> List[str]:
        """
        Filter tickers to only those with market cap >= min_market_cap.

        Args:
            tickers: List of ticker symbols
            min_market_cap: Minimum market cap threshold (default: $20B)

        Returns:
            Filtered list of tickers meeting the threshold
        """
        filtered = []
        for ticker in tickers:
            fundamentals = self.fetch_alpha_vantage_fundamentals(ticker)
            if fundamentals.get("market_cap", 0) >= min_market_cap:
                filtered.append(ticker)
                logger.info(
                    f"{ticker}: ${fundamentals['market_cap']/1e9:.2f}B market cap (included)"
                )
            else:
                logger.info(
                    f"{ticker}: ${fundamentals.get('market_cap', 0)/1e9:.2f}B market cap (excluded)"
                )
        return filtered

    @staticmethod
    def align_dataframes(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align multiple DataFrames by date index and forward-fill missing values.

        Args:
            dfs: Dictionary of DataFrames with date indices

        Returns:
            Merged DataFrame with all data aligned
        """
        merged = None
        for name, df in dfs.items():
            if merged is None:
                merged = df.rename(columns={df.columns[0]: name})
            else:
                # Merge on date index with outer join to preserve all dates
                merged = merged.join(
                    df.rename(columns={df.columns[0]: name}), how="outer"
                )

        # Forward fill for gaps, then backward fill for leading NaNs
        merged = merged.fillna(method="ffill").fillna(method="bfill")
        return merged


if __name__ == "__main__":
    # Example usage
    loader = DataLoader(
        alpha_vantage_key="demo",  # Replace with your key
        fred_api_key="demo",  # Replace with your key
    )

    # Fetch daily data
    aapl_data = loader.fetch_alpha_vantage_daily("AAPL")
    print("AAPL Daily Data:", aapl_data.head())

    # Fetch fundamentals
    fundamentals = loader.fetch_alpha_vantage_fundamentals("AAPL")
    print("AAPL Fundamentals:", fundamentals)

    # Fetch macro data
    ten_year = loader.fetch_fred_series("DGS10")
    print("10-Year Yield:", ten_year.head())
