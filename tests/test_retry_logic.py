"""
Test retry logic with exponential backoff.
"""

import pytest
import sys
import time
import logging
from unittest.mock import Mock, patch
from tenacity import RetryError

# Configure stdout to handle unicode on Windows
# Skip this when running under pytest to avoid conflicts with pytest's capture
if sys.platform == "win32" and "pytest" not in sys.modules:
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Configure logging to see retry attempts
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class TestRetryLogic:
    """Test retry logic with exponential backoff."""

    def test_fetcher_retry_with_recovery(self):
        """Test retry logic recovers from transient network errors."""
        from cache import CacheManager
        from fetcher import CachedDataFetcher
        import pandas as pd

        cache = CacheManager(cache_dir="./.cache_test", ttl_hours=24)
        fetcher = CachedDataFetcher(cache)

        call_count = {"count": 0}

        def flaky_history(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise OSError("Simulated network error")
            # Return mock DataFrame on 3rd attempt
            return pd.DataFrame(
                {
                    "Date": pd.date_range("2024-01-01", periods=10),
                    "Close": [100 + i for i in range(10)],
                    "Open": [99 + i for i in range(10)],
                    "High": [101 + i for i in range(10)],
                    "Low": [98 + i for i in range(10)],
                    "Volume": [1000000] * 10,
                }
            )

        # Mock both cache to return None and yfinance Ticker
        with patch.object(cache, "get", return_value=None), patch(
            "fetcher.yf.Ticker"
        ) as mock_ticker_class:
            mock_ticker = Mock()
            mock_ticker.history = flaky_history
            mock_ticker_class.return_value = mock_ticker

            result = fetcher.fetch_price_history("TESTRETRY", "1y")

            assert (
                call_count["count"] == 3
            ), f"Expected 3 attempts, got {call_count['count']}"
            assert len(result) > 0, "Expected non-empty DataFrame"

    def test_fetcher_retry_max_attempts(self):
        """Test retry logic fails after max attempts for persistent errors."""
        from cache import CacheManager
        from fetcher import CachedDataFetcher

        cache = CacheManager(cache_dir="./.cache_test", ttl_hours=24)
        fetcher = CachedDataFetcher(cache)

        call_count = {"count": 0}

        def always_fail(*args, **kwargs):
            call_count["count"] += 1
            raise OSError("Persistent network error")

        with patch("fetcher.yf.Ticker") as mock_ticker_class:
            mock_ticker = Mock()
            mock_ticker.history = always_fail
            mock_ticker_class.return_value = mock_ticker

            with pytest.raises(OSError, match="Persistent network error"):
                fetcher.fetch_price_history("BADTICKER", "1y")

            assert (
                call_count["count"] == 3
            ), f"Expected 3 attempts, got {call_count['count']}"

    def test_fetcher_no_retry_on_value_error(self):
        """Test that ValueError does not trigger retries."""
        from cache import CacheManager
        from fetcher import CachedDataFetcher

        cache = CacheManager(cache_dir="./.cache_test", ttl_hours=24)
        fetcher = CachedDataFetcher(cache)

        call_count = {"count": 0}

        def immediate_fail(*args, **kwargs):
            call_count["count"] += 1
            raise ValueError("Invalid ticker")

        with patch("fetcher.yf.Ticker") as mock_ticker_class:
            mock_ticker = Mock()
            mock_ticker.history = immediate_fail
            mock_ticker_class.return_value = mock_ticker

            with pytest.raises(ValueError, match="Invalid ticker"):
                fetcher.fetch_price_history("INVALID", "1y")

            assert (
                call_count["count"] == 1
            ), f"Expected 1 attempt (no retry), got {call_count['count']}"


class TestLLMRetryLogic:
    """Test LLM interface retry logic."""

    def test_llm_parse_retry_with_recovery(self):
        """Test LLM parse recovers from transient errors."""
        from config import Config
        from llm_interface import IntegratedLLMInterface

        config = Config(
            openai_api_key="test-key",
            model_name="gpt-4o",
            cache_ttl_hours=24,
            request_timeout=30,
            risk_free_rate=0.02,
            benchmark_ticker="SPY",
        )

        call_count = {"count": 0}

        def flaky_invoke(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] < 2:
                raise ConnectionError("Simulated API timeout")
            # Return mock response on 2nd attempt
            mock_response = Mock()
            mock_response.content = '{"tickers": ["AAPL", "MSFT"], "period": "1y", "metrics": [], "output_format": "markdown"}'
            return mock_response

        with patch.object(IntegratedLLMInterface, "__init__", lambda x, y: None):
            llm = IntegratedLLMInterface(config)
            llm.parser_chain = Mock()
            llm.parser_chain.invoke = flaky_invoke

            result = llm.parse_natural_language_request("Analyze AAPL and MSFT")

            assert (
                call_count["count"] == 2
            ), f"Expected 2 attempts, got {call_count['count']}"
            assert result.tickers == ["AAPL", "MSFT"], "Unexpected parse result"

    def test_llm_no_retry_on_json_error(self):
        """Test that JSON errors do not trigger retries."""
        from config import Config
        from llm_interface import IntegratedLLMInterface

        config = Config(
            openai_api_key="test-key",
            model_name="gpt-4o",
            cache_ttl_hours=24,
            request_timeout=30,
            risk_free_rate=0.02,
            benchmark_ticker="SPY",
        )

        call_count = {"count": 0}

        def bad_json(*args, **kwargs):
            call_count["count"] += 1
            mock_response = Mock()
            mock_response.content = "invalid json"
            return mock_response

        with patch.object(IntegratedLLMInterface, "__init__", lambda x, y: None):
            llm = IntegratedLLMInterface(config)
            llm.parser_chain = Mock()
            llm.parser_chain.invoke = bad_json

            with pytest.raises(ValueError):
                llm.parse_natural_language_request("Analyze AAPL")

            assert (
                call_count["count"] == 1
            ), f"Expected 1 attempt (no retry), got {call_count['count']}"


class TestExponentialBackoff:
    """Test exponential backoff timing."""

    def test_exponential_backoff_timing(self):
        """Test that exponential backoff timing is approximately correct."""
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_exponential,
            retry_if_exception_type,
        )

        call_times = []

        @retry(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(ValueError),
            reraise=True,
        )
        def failing_function():
            call_times.append(time.time())
            raise ValueError("Test error")

        try:
            failing_function()
        except (ValueError, RetryError):
            pass

        assert len(call_times) >= 2, "Expected at least 2 retry attempts"

        delays = [call_times[i] - call_times[i - 1] for i in range(1, len(call_times))]
        expected = [1, 2, 4]

        # Check delays are approximately correct (within 0.5s tolerance)
        for i, (actual, expected_delay) in enumerate(zip(delays, expected)):
            diff = abs(actual - expected_delay)
            assert (
                diff <= 0.5
            ), f"Delay {i+1} differs by {diff:.2f}s from expected {expected_delay}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
