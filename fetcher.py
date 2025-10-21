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

logger = logging.getLogger(__name__)


class CachedDataFetcher:
    """Data fetcher with intelligent caching and connection pooling."""

    def __init__(self, cache_manager: CacheManager, timeout: int = 30,
                 pool_connections: int = 10, pool_maxsize: int = 10) -> None:
        """Initialize fetcher with caching.

        Note: yfinance (v0.2.28+) uses curl_cffi internally which manages its own
        connection pooling. Custom requests.Session is no longer supported.

        Args:
            cache_manager: Cache manager instance
            timeout: Request timeout in seconds
            pool_connections: Number of connection pools (reserved for future use)
            pool_maxsize: Maximum connections per pool (reserved for future use)
        """
        self.cache = cache_manager
        self.timeout = timeout

        # Create a session for potential future non-yfinance HTTP requests
        # Note: yfinance no longer accepts custom sessions (uses curl_cffi)
        self.session = self._create_pooled_session(pool_connections, pool_maxsize)

    def _create_pooled_session(self, pool_connections: int = 10,
                               pool_maxsize: int = 10) -> Session:
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
            allowed_methods=["HEAD", "GET", "OPTIONS"]  # Retry on these methods
        )

        # Create adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=retry_strategy,
            pool_block=False  # Don't block when pool is full
        )

        # Mount adapter for both http and https
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        logger.info(f"Created HTTP session with connection pool (size: {pool_maxsize})")
        return session
    
    def fetch_price_history(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Fetch price history with caching.

        Note: yfinance now uses curl_cffi internally for rate limiting bypass,
        which doesn't support custom requests.Session. We let yfinance manage
        its own connection pooling.
        """
        ticker = ticker.strip().upper()

        # Try cache first
        cached_data = self.cache.get(ticker, period, "prices")
        if cached_data is not None:
            return cached_data

        # Fetch fresh data using yfinance's default session
        # yfinance manages its own connection pooling via curl_cffi
        logger.info(f"Fetching price history for {ticker} (period: {period})")
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period, auto_adjust=False)

            if hist.empty:
                raise ValueError(f"No data for {ticker}")

            hist = hist.reset_index()
            hist["ticker"] = ticker

            if len(hist) < 5:
                raise ValueError(f"Insufficient data for {ticker}")

            # Cache the result
            self.cache.set(ticker, period, hist, "prices")

            logger.info(f"Fetched {len(hist)} rows for {ticker}")
            return hist

        except ValueError:
            # Re-raise ValueError as-is (our own validation errors)
            raise
        except (KeyError, AttributeError, TypeError, OSError) as e:
            # Handle yfinance API errors, network issues, data parsing errors
            logger.error(f"Failed to fetch {ticker}: {e}")
            raise ValueError(f"Failed to fetch {ticker}: {str(e)}") from e
    
    def validate_ticker_exists(self, ticker: str) -> bool:
        """Quick validation to check if ticker exists.

        Note: Uses yfinance's default session (curl_cffi).
        """
        try:
            t = yf.Ticker(ticker)
            info = t.info
            return bool(info and 'symbol' in info)
        except (KeyError, AttributeError, TypeError, OSError, ValueError):
            return False

    def __del__(self) -> None:
        """Cleanup: Close the session when object is destroyed."""
        if hasattr(self, 'session'):
            try:
                self.session.close()
                logger.debug("Closed HTTP session pool")
            except Exception:
                pass  # Ignore errors during cleanup
