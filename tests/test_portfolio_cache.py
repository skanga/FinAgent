"""
Tests for PortfolioAnalyzer LRU cache functionality.

Tests cover:
- Cache initialization with custom max size
- Cache eviction when full (LRU policy)
- Cache hit moves item to end (marks as recently used)
- Cache size limit enforcement
- clear_cache() method
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from analyzers import PortfolioAnalyzer
from models import TickerAnalysis, AdvancedMetrics, TechnicalIndicators, FundamentalData
from constants import Defaults


@pytest.fixture
def portfolio_analyzer():
    """Create a PortfolioAnalyzer with small cache for testing."""
    return PortfolioAnalyzer(risk_free_rate=0.02, cache_max_size=3)


@pytest.fixture
def portfolio_analyzer_default():
    """Create a PortfolioAnalyzer with default cache size."""
    return PortfolioAnalyzer(risk_free_rate=0.02)


@pytest.fixture
def sample_ticker_analysis():
    """Create a sample TickerAnalysis for testing."""
    def _create_analysis(ticker: str):
        return TickerAnalysis(
            ticker=ticker,
            csv_path=Path(f"/tmp/{ticker}.csv"),
            chart_path=Path(f"/tmp/{ticker}.png"),
            latest_close=100.0,
            avg_daily_return=0.001,
            volatility=0.02,
            ratios={},
            fundamentals=FundamentalData(),
            advanced_metrics=AdvancedMetrics(),
            technical_indicators=TechnicalIndicators(),
        )
    return _create_analysis


class TestLRUCacheInitialization:
    """Test LRU cache initialization."""

    def test_default_cache_size(self, portfolio_analyzer_default):
        """Test that default cache size is set from constants."""
        assert portfolio_analyzer_default._cache_max_size == Defaults.PORTFOLIO_CACHE_MAX_SIZE
        assert portfolio_analyzer_default._cache_max_size == 100

    def test_custom_cache_size(self):
        """Test custom cache size initialization."""
        analyzer = PortfolioAnalyzer(cache_max_size=50)
        assert analyzer._cache_max_size == 50

    def test_cache_is_ordered_dict(self, portfolio_analyzer):
        """Test that cache is OrderedDict for LRU behavior."""
        from collections import OrderedDict
        assert isinstance(portfolio_analyzer._returns_cache, OrderedDict)

    def test_cache_starts_empty(self, portfolio_analyzer):
        """Test that cache starts empty."""
        assert len(portfolio_analyzer._returns_cache) == 0


class TestCacheEviction:
    """Test LRU cache eviction policy."""

    def test_cache_evicts_oldest_when_full(self, portfolio_analyzer, sample_ticker_analysis):
        """Test that oldest entry is evicted when cache is full."""
        # Create sample analyses (cache_max_size = 3)
        analyses_1 = {
            "AAPL": sample_ticker_analysis("AAPL"),
            "MSFT": sample_ticker_analysis("MSFT"),
        }
        analyses_2 = {
            "GOOGL": sample_ticker_analysis("GOOGL"),
            "AMZN": sample_ticker_analysis("AMZN"),
        }
        analyses_3 = {
            "TSLA": sample_ticker_analysis("TSLA"),
            "NVDA": sample_ticker_analysis("NVDA"),
        }
        analyses_4 = {
            "META": sample_ticker_analysis("META"),
            "NFLX": sample_ticker_analysis("NFLX"),
        }

        # Mock CSV reads
        mock_df = pd.DataFrame({"daily_return": [0.01, 0.02, 0.015, -0.01, 0.005] * 10})

        with patch("pandas.read_csv", return_value=mock_df):
            # Fill cache to max (3 entries)
            portfolio_analyzer._load_returns_data(analyses_1)
            portfolio_analyzer._load_returns_data(analyses_2)
            portfolio_analyzer._load_returns_data(analyses_3)

            assert len(portfolio_analyzer._returns_cache) == 3

            # Add 4th entry - should evict oldest (analyses_1)
            portfolio_analyzer._load_returns_data(analyses_4)

            assert len(portfolio_analyzer._returns_cache) == 3
            # Check that analyses_1 was evicted
            assert frozenset(analyses_1.keys()) not in portfolio_analyzer._returns_cache
            # Check that newest entries are present
            assert frozenset(analyses_2.keys()) in portfolio_analyzer._returns_cache
            assert frozenset(analyses_3.keys()) in portfolio_analyzer._returns_cache
            assert frozenset(analyses_4.keys()) in portfolio_analyzer._returns_cache

    def test_cache_never_exceeds_max_size(self, portfolio_analyzer, sample_ticker_analysis):
        """Test that cache never exceeds max_size."""
        mock_df = pd.DataFrame({"daily_return": [0.01, 0.02, 0.015] * 10})

        with patch("pandas.read_csv", return_value=mock_df):
            # Add 10 different portfolio combinations
            for i in range(10):
                analyses = {
                    f"TICK{i}A": sample_ticker_analysis(f"TICK{i}A"),
                    f"TICK{i}B": sample_ticker_analysis(f"TICK{i}B"),
                }
                portfolio_analyzer._load_returns_data(analyses)

                # Cache should never exceed 3
                assert len(portfolio_analyzer._returns_cache) <= 3


class TestCacheLRUBehavior:
    """Test LRU (Least Recently Used) behavior."""

    def test_cache_hit_moves_to_end(self, portfolio_analyzer, sample_ticker_analysis):
        """Test that cache hit moves entry to end (marks as recently used)."""
        analyses_1 = {
            "AAPL": sample_ticker_analysis("AAPL"),
            "MSFT": sample_ticker_analysis("MSFT"),
        }
        analyses_2 = {
            "GOOGL": sample_ticker_analysis("GOOGL"),
            "AMZN": sample_ticker_analysis("AMZN"),
        }
        analyses_3 = {
            "TSLA": sample_ticker_analysis("TSLA"),
            "NVDA": sample_ticker_analysis("NVDA"),
        }
        analyses_4 = {
            "META": sample_ticker_analysis("META"),
            "NFLX": sample_ticker_analysis("NFLX"),
        }

        mock_df = pd.DataFrame({"daily_return": [0.01, 0.02, 0.015] * 10})

        with patch("pandas.read_csv", return_value=mock_df):
            # Fill cache: [1, 2, 3]
            portfolio_analyzer._load_returns_data(analyses_1)
            portfolio_analyzer._load_returns_data(analyses_2)
            portfolio_analyzer._load_returns_data(analyses_3)

            # Access analyses_1 (oldest) - should move to end: [2, 3, 1]
            portfolio_analyzer._load_returns_data(analyses_1)

            # Add analyses_4 - should evict analyses_2 (now oldest): [3, 1, 4]
            portfolio_analyzer._load_returns_data(analyses_4)

            # Check that analyses_1 is still in cache (was moved to end)
            assert frozenset(analyses_1.keys()) in portfolio_analyzer._returns_cache
            # Check that analyses_2 was evicted
            assert frozenset(analyses_2.keys()) not in portfolio_analyzer._returns_cache
            # Check that newest entries are present
            assert frozenset(analyses_3.keys()) in portfolio_analyzer._returns_cache
            assert frozenset(analyses_4.keys()) in portfolio_analyzer._returns_cache

    def test_multiple_cache_hits_preserve_order(self, portfolio_analyzer, sample_ticker_analysis):
        """Test that multiple cache hits correctly update LRU order."""
        analyses_1 = {"A": sample_ticker_analysis("A"), "B": sample_ticker_analysis("B")}
        analyses_2 = {"C": sample_ticker_analysis("C"), "D": sample_ticker_analysis("D")}
        analyses_3 = {"E": sample_ticker_analysis("E"), "F": sample_ticker_analysis("F")}

        mock_df = pd.DataFrame({"daily_return": [0.01, 0.02] * 10})

        with patch("pandas.read_csv", return_value=mock_df):
            # Fill cache: [1, 2, 3]
            portfolio_analyzer._load_returns_data(analyses_1)
            portfolio_analyzer._load_returns_data(analyses_2)
            portfolio_analyzer._load_returns_data(analyses_3)

            # Access in order: 2, 1, 3 -> new order: [2, 1, 3]
            portfolio_analyzer._load_returns_data(analyses_2)
            portfolio_analyzer._load_returns_data(analyses_1)
            portfolio_analyzer._load_returns_data(analyses_3)

            # Verify order by checking keys
            keys = list(portfolio_analyzer._returns_cache.keys())
            assert keys[0] == frozenset(analyses_2.keys())
            assert keys[1] == frozenset(analyses_1.keys())
            assert keys[2] == frozenset(analyses_3.keys())


class TestCacheHitAndMiss:
    """Test cache hit and miss behavior."""

    def test_cache_hit_returns_same_dataframe(self, portfolio_analyzer, sample_ticker_analysis):
        """Test that cache hit returns the exact same DataFrame."""
        analyses = {
            "AAPL": sample_ticker_analysis("AAPL"),
            "MSFT": sample_ticker_analysis("MSFT"),
        }

        mock_df = pd.DataFrame({"daily_return": [0.01, 0.02, 0.015] * 10})

        with patch("pandas.read_csv", return_value=mock_df):
            # First call - cache miss
            result1 = portfolio_analyzer._load_returns_data(analyses)

            # Second call - cache hit
            result2 = portfolio_analyzer._load_returns_data(analyses)

            # Should return same DataFrame object
            assert result1 is result2
            pd.testing.assert_frame_equal(result1, result2)

    def test_cache_miss_loads_from_csv(self, portfolio_analyzer, sample_ticker_analysis):
        """Test that cache miss loads data from CSV."""
        analyses = {
            "AAPL": sample_ticker_analysis("AAPL"),
            "MSFT": sample_ticker_analysis("MSFT"),
        }

        mock_df = pd.DataFrame({"daily_return": [0.01, 0.02, 0.015] * 10})

        with patch("pandas.read_csv", return_value=mock_df) as mock_read:
            # First call - should read CSV
            portfolio_analyzer._load_returns_data(analyses)
            assert mock_read.call_count == 2  # One for each ticker

            # Second call - should use cache
            mock_read.reset_mock()
            portfolio_analyzer._load_returns_data(analyses)
            assert mock_read.call_count == 0  # No CSV reads


class TestCacheClear:
    """Test cache clearing functionality."""

    def test_clear_cache(self, portfolio_analyzer, sample_ticker_analysis):
        """Test that clear_cache empties the cache."""
        analyses = {
            "AAPL": sample_ticker_analysis("AAPL"),
            "MSFT": sample_ticker_analysis("MSFT"),
        }

        mock_df = pd.DataFrame({"daily_return": [0.01, 0.02] * 10})

        with patch("pandas.read_csv", return_value=mock_df):
            # Add to cache
            portfolio_analyzer._load_returns_data(analyses)
            assert len(portfolio_analyzer._returns_cache) == 1

            # Clear cache
            portfolio_analyzer.clear_cache()
            assert len(portfolio_analyzer._returns_cache) == 0

    def test_clear_cache_allows_reloading(self, portfolio_analyzer, sample_ticker_analysis):
        """Test that data can be reloaded after clearing cache."""
        analyses = {
            "AAPL": sample_ticker_analysis("AAPL"),
            "MSFT": sample_ticker_analysis("MSFT"),
        }

        mock_df = pd.DataFrame({"daily_return": [0.01, 0.02] * 10})

        with patch("pandas.read_csv", return_value=mock_df) as mock_read:
            # Load data
            portfolio_analyzer._load_returns_data(analyses)
            assert mock_read.call_count == 2

            # Clear cache
            portfolio_analyzer.clear_cache()

            # Load again - should read CSV
            mock_read.reset_mock()
            portfolio_analyzer._load_returns_data(analyses)
            assert mock_read.call_count == 2


class TestCacheWithDifferentPortfolios:
    """Test cache behavior with different portfolio combinations."""

    def test_different_tickers_different_cache_keys(self, portfolio_analyzer, sample_ticker_analysis):
        """Test that different ticker combinations use different cache keys."""
        analyses_1 = {
            "AAPL": sample_ticker_analysis("AAPL"),
            "MSFT": sample_ticker_analysis("MSFT"),
        }
        analyses_2 = {
            "AAPL": sample_ticker_analysis("AAPL"),
            "GOOGL": sample_ticker_analysis("GOOGL"),
        }

        mock_df = pd.DataFrame({"daily_return": [0.01, 0.02] * 10})

        with patch("pandas.read_csv", return_value=mock_df):
            portfolio_analyzer._load_returns_data(analyses_1)
            portfolio_analyzer._load_returns_data(analyses_2)

            # Both should be cached separately
            assert len(portfolio_analyzer._returns_cache) == 2
            assert frozenset(analyses_1.keys()) in portfolio_analyzer._returns_cache
            assert frozenset(analyses_2.keys()) in portfolio_analyzer._returns_cache

    def test_same_tickers_same_cache_key(self, portfolio_analyzer, sample_ticker_analysis):
        """Test that same tickers use same cache key."""
        analyses_1 = {
            "AAPL": sample_ticker_analysis("AAPL"),
            "MSFT": sample_ticker_analysis("MSFT"),
        }
        # Different order, same tickers
        analyses_2 = {
            "MSFT": sample_ticker_analysis("MSFT"),
            "AAPL": sample_ticker_analysis("AAPL"),
        }

        mock_df = pd.DataFrame({"daily_return": [0.01, 0.02] * 10})

        with patch("pandas.read_csv", return_value=mock_df) as mock_read:
            portfolio_analyzer._load_returns_data(analyses_1)
            assert mock_read.call_count == 2

            # Second call with same tickers (different order) - should hit cache
            mock_read.reset_mock()
            portfolio_analyzer._load_returns_data(analyses_2)
            assert mock_read.call_count == 0  # Cache hit

            # Only one cache entry
            assert len(portfolio_analyzer._returns_cache) == 1


class TestCacheEdgeCases:
    """Test edge cases for cache functionality."""

    def test_cache_with_single_ticker(self, portfolio_analyzer, sample_ticker_analysis):
        """Test that single-ticker portfolios return None (need 2+)."""
        analyses = {"AAPL": sample_ticker_analysis("AAPL")}

        mock_df = pd.DataFrame({"daily_return": [0.01, 0.02] * 10})

        with patch("pandas.read_csv", return_value=mock_df):
            result = portfolio_analyzer._load_returns_data(analyses)
            # Should return None (need at least 2 tickers)
            assert result is None
            # Should not be cached
            assert len(portfolio_analyzer._returns_cache) == 0

    def test_cache_with_csv_read_errors(self, portfolio_analyzer, sample_ticker_analysis):
        """Test cache behavior when CSV reads fail."""
        analyses = {
            "AAPL": sample_ticker_analysis("AAPL"),
            "MSFT": sample_ticker_analysis("MSFT"),
        }

        # Mock CSV read to fail
        with patch("pandas.read_csv", side_effect=OSError("File not found")):
            result = portfolio_analyzer._load_returns_data(analyses)
            # Should return None when data loading fails
            assert result is None
            # Should not be cached
            assert len(portfolio_analyzer._returns_cache) == 0

    def test_cache_size_one(self):
        """Test cache with max_size=1."""
        analyzer = PortfolioAnalyzer(cache_max_size=1)

        def create_analysis(ticker):
            return TickerAnalysis(
                ticker=ticker,
                csv_path=Path(f"/tmp/{ticker}.csv"),
                chart_path=Path(f"/tmp/{ticker}.png"),
                latest_close=100.0,
                avg_daily_return=0.001,
                volatility=0.02,
                ratios={},
                fundamentals=FundamentalData(),
                advanced_metrics=AdvancedMetrics(),
                technical_indicators=TechnicalIndicators(),
            )

        analyses_1 = {"A": create_analysis("A"), "B": create_analysis("B")}
        analyses_2 = {"C": create_analysis("C"), "D": create_analysis("D")}

        mock_df = pd.DataFrame({"daily_return": [0.01, 0.02] * 10})

        with patch("pandas.read_csv", return_value=mock_df):
            analyzer._load_returns_data(analyses_1)
            assert len(analyzer._returns_cache) == 1

            analyzer._load_returns_data(analyses_2)
            # Should evict first entry
            assert len(analyzer._returns_cache) == 1
            assert frozenset(analyses_2.keys()) in analyzer._returns_cache
            assert frozenset(analyses_1.keys()) not in analyzer._returns_cache
