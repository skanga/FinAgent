"""
Data fetching with caching support.
"""

import logging
import pandas as pd
import yfinance as yf
from requests import Session
from cache import CacheManager
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from utils import validate_ticker_symbol
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


class CachedDataFetcher:
    """Data fetcher with intelligent caching and connection pooling."""

    def __init__(
        self,
        cache_manager: CacheManager,
        timeout: int = 30,
        pool_connections: int = 10,
        pool_maxsize: int = 10,
    ) -> None:
        """
        Initializes the CachedDataFetcher with a cache manager and connection pooling settings.

        Args:
            cache_manager (CacheManager): The cache manager to use.
            timeout (int): The request timeout in seconds.
            pool_connections (int): The number of connection pools.
            pool_maxsize (int): The maximum number of connections per pool.
        """
        self.cache = cache_manager
        self.timeout = timeout

        # Create a session for potential future non-yfinance HTTP requests
        # Note: yfinance no longer accepts custom sessions (uses curl_cffi)
        self.session = self._create_pooled_session(pool_connections, pool_maxsize)

    def _create_pooled_session(
        self, pool_connections: int = 10, pool_maxsize: int = 10
    ) -> Session:
        """Create a requests Session with connection pooling and retry logic.

        Args:
            pool_connections: Number of connection pools
            pool_maxsize: Max connections per pool

        Returns:
            Configured Session object
        """
        session = Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,  # Total retries
            backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["HEAD", "GET", "OPTIONS"],  # Retry on these methods
        )

        # Create adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=retry_strategy,
            pool_block=False,  # Don't block when pool is full
        )

        # Mount adapter for both http and https
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

        logger.debug(f"Created HTTP session with connection pool (size: {pool_maxsize})")
        return session

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((OSError, ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def fetch_price_history(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetches the price history for a given ticker, using the cache if available.

        Automatically retries up to 3 times with exponential backoff (1s, 2s, 4s)
        for transient network errors (OSError, ConnectionError, TimeoutError).

        Args:
            ticker (str): The ticker symbol to fetch.
            period (str): The period to fetch the history for.

        Returns:
            pd.DataFrame: A DataFrame containing the price history.

        Raises:
            ValueError: If ticker is invalid or has insufficient data
            OSError: If network errors persist after retries
        """
        # Validate and normalize ticker using centralized validation
        ticker = validate_ticker_symbol(ticker)

        # Try cache first
        cached_data = self.cache.get(ticker, period, "prices")
        if cached_data is not None:
            return cached_data

        # Fetch fresh data using yfinance's default session
        # yfinance manages its own connection pooling via curl_cffi
        logger.debug(f"Fetching price history for {ticker} (period: {period})")
        try:
            ticker_obj = yf.Ticker(ticker)
            price_history = ticker_obj.history(period=period, auto_adjust=False)

            if price_history.empty:
                raise ValueError(f"No data for {ticker}")

            price_history = price_history.reset_index()
            price_history["ticker"] = ticker

            if len(price_history) < 5:
                raise ValueError(f"Insufficient data for {ticker}")

            # Cache the result
            self.cache.set(ticker, period, price_history, "prices")

            logger.debug(f"Fetched {len(price_history)} rows for {ticker}")
            return price_history

        except ValueError:
            # Re-raise ValueError as-is (our own validation errors)
            # Don't retry for invalid tickers or data issues
            raise
        except OSError:
            # Re-raise OSError to trigger retry
            logger.error(f"Network error fetching {ticker}")
            raise
        except (KeyError, AttributeError, TypeError) as e:
            # Handle yfinance API errors and data parsing errors
            # These don't trigger retry
            logger.error(f"Failed to fetch {ticker}: {e}")
            raise ValueError(f"Failed to fetch {ticker}: {str(e)}") from e

    def validate_ticker_exists(self, ticker: str) -> bool:
        """
        Validates that a ticker exists by checking for its info.

        Args:
            ticker (str): The ticker symbol to validate.

        Returns:
            bool: True if the ticker exists, False otherwise.
        """
        try:
            t = yf.Ticker(ticker)
            info = t.info
            return bool(info and "symbol" in info)
        except (KeyError, AttributeError, TypeError, OSError, ValueError):
            return False

    def __del__(self) -> None:
        """Cleanup: Close the session when object is destroyed."""
        if hasattr(self, "session"):
            try:
                self.session.close()
                logger.debug("Closed HTTP session pool")
            except Exception:
                pass  # Ignore errors during cleanup
