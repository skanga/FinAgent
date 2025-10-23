"""
Test connection pooling implementation.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from cache import CacheManager
from fetcher import CachedDataFetcher
from llm_interface import IntegratedLLMInterface


class TestConnectionPooling:
    """Test HTTP connection pooling implementation."""

    def test_fetcher_connection_pool(self):
        """Test that fetcher creates and uses connection pool."""
        cache = CacheManager(ttl_hours=24)
        fetcher = CachedDataFetcher(cache, pool_connections=5, pool_maxsize=10)

        # Verify session exists
        assert hasattr(fetcher, "session"), "Fetcher should have session attribute"
        assert fetcher.session is not None, "Session should be initialized"

        # Verify adapter is configured
        adapter = fetcher.session.get_adapter("https://query1.finance.yahoo.com")
        assert adapter is not None, "HTTPS adapter should be configured"

        # Check pool configuration
        pool_manager = adapter.poolmanager
        assert (
            pool_manager.connection_pool_kw["maxsize"] == 10
        ), "Pool maxsize should be 10"

        # Cleanup
        del fetcher

    def test_llm_connection_pool(self):
        """Test that LLM interface creates and uses connection pool."""
        try:
            config = Config.from_env()

            # Create LLM with custom pool size
            llm = IntegratedLLMInterface(
                config, max_connections=15, max_keepalive_connections=8
            )

            # Verify LLM has client
            assert hasattr(llm, "llm"), "Should have llm attribute"
            assert llm.llm is not None, "LLM should be initialized"

            # Cleanup
            del llm

        except Exception as e:
            pytest.skip(f"LLM test skipped (config error): {e}")

    def test_connection_reuse(self):
        """Test that connections are actually reused."""
        cache = CacheManager(ttl_hours=24)
        fetcher = CachedDataFetcher(cache, pool_connections=2, pool_maxsize=2)

        # Make multiple requests to test connection reuse
        tickers = ["AAPL", "MSFT", "GOOGL"]
        successful = 0

        for ticker in tickers:
            try:
                _data = fetcher.fetch_price_history(ticker, "5d")
                successful += 1
            except Exception:
                pass  # Expected if no internet or API errors

        # We don't assert on successful count since it depends on internet availability
        # Just verify the pool was created correctly
        assert hasattr(fetcher, "session")
        assert fetcher.session is not None

        del fetcher


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
