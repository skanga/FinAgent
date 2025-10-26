"""
Comprehensive tests for data fetcher with caching.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from cache import CacheManager
from fetcher import CachedDataFetcher


@pytest.fixture
def cache_manager(tmp_path):
    """Create a cache manager for testing."""
    cache_dir = tmp_path / ".test_cache"
    return CacheManager(str(cache_dir), ttl_hours=1)


@pytest.fixture
def fetcher(cache_manager):
    """Create a cached data fetcher for testing."""
    return CachedDataFetcher(cache_manager, timeout=10)


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=252, freq="D")
    return pd.DataFrame({
        "Date": dates,
        "Open": [100.0 + i * 0.5 for i in range(252)],
        "High": [102.0 + i * 0.5 for i in range(252)],
        "Low": [98.0 + i * 0.5 for i in range(252)],
        "Close": [101.0 + i * 0.5 for i in range(252)],
        "Volume": [1000000 + i * 1000 for i in range(252)],
        "Dividends": [0.0] * 252,
        "Stock Splits": [0.0] * 252,
    })


class TestFetcherInitialization:
    """Test fetcher initialization."""

    def test_initialization(self, cache_manager):
        """Test successful initialization."""
        fetcher = CachedDataFetcher(cache_manager)

        assert fetcher.cache == cache_manager
        assert fetcher.timeout == 30
        assert hasattr(fetcher, "session")

    def test_custom_timeout(self, cache_manager):
        """Test initialization with custom timeout."""
        fetcher = CachedDataFetcher(cache_manager, timeout=60)

        assert fetcher.timeout == 60

    def test_custom_pool_settings(self, cache_manager):
        """Test initialization with custom pool settings."""
        fetcher = CachedDataFetcher(
            cache_manager,
            pool_connections=20,
            pool_maxsize=20
        )

        assert fetcher is not None

    def test_session_created(self, fetcher):
        """Test that session is properly created."""
        assert fetcher.session is not None
        assert hasattr(fetcher.session, "headers")
        assert "User-Agent" in fetcher.session.headers


class TestSessionConfiguration:
    """Test session and connection pool configuration."""

    def test_retry_strategy(self, fetcher):
        """Test that retry strategy is configured."""
        adapter = fetcher.session.get_adapter("https://example.com")

        assert adapter is not None
        assert hasattr(adapter, "max_retries")
        assert adapter.max_retries.total == 3

    def test_connection_pooling(self, fetcher):
        """Test connection pool configuration."""
        adapter = fetcher.session.get_adapter("https://example.com")

        assert adapter.poolmanager is None or hasattr(adapter, "_pool_connections")

    def test_user_agent_header(self, fetcher):
        """Test that User-Agent header is set."""
        assert "User-Agent" in fetcher.session.headers
        assert "Mozilla" in fetcher.session.headers["User-Agent"]


class TestPriceHistoryFetching:
    """Test price history fetching."""

    def test_fetch_from_cache(self, fetcher, cache_manager, sample_price_data):
        """Test fetching from cache when data is available."""
        # Pre-populate cache
        cache_manager.set("AAPL", "1y", sample_price_data, "prices")

        # Fetch should return cached data
        result = fetcher.fetch_price_history("AAPL", "1y")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_price_data)
        assert "Close" in result.columns

    def test_fetch_fresh_data(self, fetcher, sample_price_data):
        """Test fetching fresh data when cache is empty."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            # Mock yfinance Ticker
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = sample_price_data.copy()
            mock_ticker.return_value = mock_ticker_instance

            result = fetcher.fetch_price_history("MSFT", "1y")

            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert "ticker" in result.columns
            assert result["ticker"].iloc[0] == "MSFT"

    def test_ticker_normalization(self, fetcher, sample_price_data):
        """Test that tickers are normalized (uppercase, trimmed)."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = sample_price_data.copy()
            mock_ticker.return_value = mock_ticker_instance

            # Test with lowercase and whitespace
            result = fetcher.fetch_price_history("  aapl  ", "1y")

            assert result["ticker"].iloc[0] == "AAPL"

    def test_empty_data_raises_error(self, fetcher):
        """Test that empty data raises ValueError."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = pd.DataFrame()
            mock_ticker.return_value = mock_ticker_instance

            with pytest.raises(ValueError, match="No data for"):
                fetcher.fetch_price_history("INVALID", "1y")

    def test_insufficient_data_raises_error(self, fetcher):
        """Test that insufficient data raises ValueError."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            # Return only 3 rows
            insufficient_data = pd.DataFrame({
                "Date": pd.date_range("2024-01-01", periods=3),
                "Close": [100, 101, 102]
            })

            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = insufficient_data
            mock_ticker.return_value = mock_ticker_instance

            with pytest.raises(ValueError, match="Insufficient data for"):
                fetcher.fetch_price_history("AAPL", "1y")

    def test_data_is_cached_after_fetch(self, fetcher, cache_manager, sample_price_data):
        """Test that fetched data is cached."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = sample_price_data.copy()
            mock_ticker.return_value = mock_ticker_instance

            # Fetch data (should cache it)
            fetcher.fetch_price_history("GOOGL", "1y")

            # Verify data is in cache
            cached = cache_manager.get("GOOGL", "1y", "prices")
            assert cached is not None
            assert len(cached) > 0


class TestRetryLogic:
    """Test retry logic for network failures."""

    def test_retry_on_oserror(self, fetcher, sample_price_data):
        """Test that OSError triggers retry."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()

            # Fail twice, succeed on third attempt
            mock_ticker_instance.history.side_effect = [
                OSError("Network error"),
                OSError("Network error"),
                sample_price_data.copy()
            ]
            mock_ticker.return_value = mock_ticker_instance

            result = fetcher.fetch_price_history("AAPL", "1y")

            assert isinstance(result, pd.DataFrame)
            assert mock_ticker_instance.history.call_count == 3

    def test_retry_on_connection_error(self, fetcher, sample_price_data):
        """Test that ConnectionError triggers retry."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()

            # Fail once, succeed on second attempt
            mock_ticker_instance.history.side_effect = [
                ConnectionError("Connection failed"),
                sample_price_data.copy()
            ]
            mock_ticker.return_value = mock_ticker_instance

            result = fetcher.fetch_price_history("AAPL", "1y")

            assert isinstance(result, pd.DataFrame)
            assert mock_ticker_instance.history.call_count == 2

    def test_retry_on_timeout_error(self, fetcher, sample_price_data):
        """Test that TimeoutError triggers retry."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()

            # Fail once, succeed on second attempt
            mock_ticker_instance.history.side_effect = [
                TimeoutError("Request timed out"),
                sample_price_data.copy()
            ]
            mock_ticker.return_value = mock_ticker_instance

            result = fetcher.fetch_price_history("AAPL", "1y")

            assert isinstance(result, pd.DataFrame)

    def test_max_retries_exceeded(self, fetcher):
        """Test that max retries are respected."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()

            # Always fail
            mock_ticker_instance.history.side_effect = OSError("Persistent network error")
            mock_ticker.return_value = mock_ticker_instance

            with pytest.raises(OSError, match="Persistent network error"):
                fetcher.fetch_price_history("AAPL", "1y")

            # Should have tried 3 times (initial + 2 retries)
            assert mock_ticker_instance.history.call_count >= 3

    def test_no_retry_on_value_error(self, fetcher):
        """Test that ValueError does not trigger retry."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()

            # Return empty data (triggers ValueError)
            mock_ticker_instance.history.return_value = pd.DataFrame()
            mock_ticker.return_value = mock_ticker_instance

            with pytest.raises(ValueError):
                fetcher.fetch_price_history("INVALID", "1y")

            # Should only try once (no retries for ValueError)
            assert mock_ticker_instance.history.call_count == 1


class TestAPIErrors:
    """Test handling of various API errors."""

    def test_keyerror_handling(self, fetcher):
        """Test handling of KeyError from yfinance."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.side_effect = KeyError("Missing column")
            mock_ticker.return_value = mock_ticker_instance

            with pytest.raises(ValueError, match="Failed to fetch"):
                fetcher.fetch_price_history("AAPL", "1y")

    def test_attribute_error_handling(self, fetcher):
        """Test handling of AttributeError from yfinance."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.side_effect = AttributeError("Missing attribute")
            mock_ticker.return_value = mock_ticker_instance

            with pytest.raises(ValueError, match="Failed to fetch"):
                fetcher.fetch_price_history("AAPL", "1y")

    def test_type_error_handling(self, fetcher):
        """Test handling of TypeError from yfinance."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.side_effect = TypeError("Invalid type")
            mock_ticker.return_value = mock_ticker_instance

            with pytest.raises(ValueError, match="Failed to fetch"):
                fetcher.fetch_price_history("AAPL", "1y")


class TestTickerValidation:
    """Test ticker validation functionality."""

    def test_valid_ticker(self, fetcher):
        """Test validation of valid ticker."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = {"symbol": "AAPL", "longName": "Apple Inc."}
            mock_ticker.return_value = mock_ticker_instance

            result = fetcher.validate_ticker_exists("AAPL")

            assert result is True

    def test_invalid_ticker(self, fetcher):
        """Test validation of invalid ticker."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = {}
            mock_ticker.return_value = mock_ticker_instance

            result = fetcher.validate_ticker_exists("INVALID")

            assert result is False

    def test_ticker_validation_handles_errors(self, fetcher):
        """Test that validation handles errors gracefully."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = None
            mock_ticker.return_value = mock_ticker_instance

            result = fetcher.validate_ticker_exists("AAPL")

            assert result is False

    def test_ticker_validation_handles_exceptions(self, fetcher):
        """Test that validation handles exceptions gracefully."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = OSError("Network error")

            result = fetcher.validate_ticker_exists("AAPL")

            assert result is False


class TestCacheIntegration:
    """Test cache integration."""

    def test_cache_hit_avoids_api_call(self, fetcher, cache_manager, sample_price_data):
        """Test that cache hit avoids API call."""
        # Pre-populate cache
        cache_manager.set("AAPL", "1y", sample_price_data, "prices")

        with patch("fetcher.yf.Ticker") as mock_ticker:
            # This should not be called if cache works
            mock_ticker_instance = Mock()
            mock_ticker.return_value = mock_ticker_instance

            result = fetcher.fetch_price_history("AAPL", "1y")

            # Verify ticker was not called (cache hit)
            assert not mock_ticker.called
            assert isinstance(result, pd.DataFrame)

    def test_cache_miss_triggers_api_call(self, fetcher, cache_manager, sample_price_data):
        """Test that cache miss triggers API call."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = sample_price_data.copy()
            mock_ticker.return_value = mock_ticker_instance

            result = fetcher.fetch_price_history("MSFT", "1y")

            # Verify ticker was called (cache miss)
            assert mock_ticker.called
            assert isinstance(result, pd.DataFrame)

    def test_different_periods_cached_separately(self, fetcher, cache_manager, sample_price_data):
        """Test that different periods are cached separately."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = sample_price_data.copy()
            mock_ticker.return_value = mock_ticker_instance

            # Fetch for two different periods
            fetcher.fetch_price_history("AAPL", "1y")
            fetcher.fetch_price_history("AAPL", "6mo")

            # Verify both were called (different cache keys)
            assert mock_ticker_instance.history.call_count == 2

    def test_cache_stores_ticker_column(self, fetcher, cache_manager, sample_price_data):
        """Test that cached data includes ticker column."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = sample_price_data.copy()
            mock_ticker.return_value = mock_ticker_instance

            # Fetch and cache
            fetcher.fetch_price_history("NVDA", "1y")

            # Retrieve from cache
            cached = cache_manager.get("NVDA", "1y", "prices")

            assert "ticker" in cached.columns
            assert cached["ticker"].iloc[0] == "NVDA"


class TestResourceCleanup:
    """Test resource cleanup."""

    def test_session_cleanup_on_delete(self, cache_manager):
        """Test that session is closed on deletion."""
        fetcher = CachedDataFetcher(cache_manager)

        # Mock the session close method
        mock_close = Mock()
        fetcher.session.close = mock_close

        # Delete fetcher
        del fetcher

        # Verify close was called
        mock_close.assert_called_once()

    def test_cleanup_handles_missing_session(self, cache_manager):
        """Test that cleanup handles missing session gracefully."""
        fetcher = CachedDataFetcher(cache_manager)

        # Remove session attribute
        delattr(fetcher, "session")

        # Should not raise exception
        try:
            del fetcher
        except Exception as e:
            pytest.fail(f"Cleanup raised exception: {e}")


class TestEdgeCases:
    """Test edge cases."""

    def test_whitespace_in_ticker(self, fetcher, sample_price_data):
        """Test handling of whitespace in ticker."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = sample_price_data.copy()
            mock_ticker.return_value = mock_ticker_instance

            result = fetcher.fetch_price_history("  AAPL  ", "1y")

            assert result["ticker"].iloc[0] == "AAPL"

    def test_lowercase_ticker(self, fetcher, sample_price_data):
        """Test handling of lowercase ticker."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = sample_price_data.copy()
            mock_ticker.return_value = mock_ticker_instance

            result = fetcher.fetch_price_history("aapl", "1y")

            assert result["ticker"].iloc[0] == "AAPL"

    def test_mixed_case_ticker(self, fetcher, sample_price_data):
        """Test handling of mixed case ticker."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = sample_price_data.copy()
            mock_ticker.return_value = mock_ticker_instance

            result = fetcher.fetch_price_history("AaPl", "1y")

            assert result["ticker"].iloc[0] == "AAPL"


class TestTickerFormatValidation:
    """Test ticker format validation using centralized validation."""

    def test_invalid_characters_rejected(self, fetcher):
        """Test that tickers with invalid characters are rejected."""
        with pytest.raises(ValueError, match="Invalid characters"):
            fetcher.fetch_price_history("AA<script>", "1y")

    def test_ticker_too_long_rejected(self, fetcher):
        """Test that tickers longer than 10 characters are rejected."""
        with pytest.raises(ValueError, match="invalid length"):
            fetcher.fetch_price_history("VERYLONGTICKER123", "1y")

    def test_empty_ticker_rejected(self, fetcher):
        """Test that empty tickers are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            fetcher.fetch_price_history("", "1y")

    def test_sql_injection_rejected(self, fetcher):
        """Test that SQL injection attempts are rejected."""
        with pytest.raises(ValueError, match="Invalid characters|suspicious|invalid length"):
            fetcher.fetch_price_history("AAL' OR '1'='1", "1y")

    def test_xss_attempt_rejected(self, fetcher):
        """Test that XSS attempts are rejected."""
        with pytest.raises(ValueError, match="Invalid characters|invalid length"):
            fetcher.fetch_price_history("AA<script>alert(1)</script>", "1y")

    def test_special_characters_rejected(self, fetcher):
        """Test that special characters are rejected."""
        with pytest.raises(ValueError, match="Invalid characters"):
            fetcher.fetch_price_history("AA$$$", "1y")

    def test_valid_ticker_with_dot(self, fetcher, sample_price_data):
        """Test that tickers with dots are allowed (e.g., BRK.B)."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = sample_price_data.copy()
            mock_ticker.return_value = mock_ticker_instance

            result = fetcher.fetch_price_history("BRK.B", "1y")

            assert result["ticker"].iloc[0] == "BRK.B"

    def test_valid_ticker_with_hyphen(self, fetcher, sample_price_data):
        """Test that tickers with hyphens are allowed."""
        with patch("fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = sample_price_data.copy()
            mock_ticker.return_value = mock_ticker_instance

            result = fetcher.fetch_price_history("CL-C", "1y")

            assert result["ticker"].iloc[0] == "CL-C"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
