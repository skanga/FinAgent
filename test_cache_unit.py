"""
Unit tests for cache.py module.

Tests cache storage, retrieval, expiration, and error handling
with various data types and edge cases.
"""

import pytest
import pandas as pd
import time
import tempfile
import shutil
from pathlib import Path

from cache import CacheManager


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def cache_manager(temp_cache_dir):
    """Create a CacheManager instance with temporary directory."""
    return CacheManager(cache_dir=temp_cache_dir, ttl_hours=1)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for caching."""
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=10),
            "Close": [100, 102, 101, 103, 105, 104, 106, 108, 107, 109],
            "Volume": [1000000] * 10,
        }
    )


class TestCacheInitialization:
    """Test CacheManager initialization."""

    def test_cache_manager_creates_directory(self, temp_cache_dir):
        """Test that cache directory is created if it doesn't exist."""
        cache_dir = Path(temp_cache_dir) / "new_cache"
        assert not cache_dir.exists()

        _cache_manager = CacheManager(cache_dir=str(cache_dir), ttl_hours=24)

        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_cache_manager_default_ttl(self, temp_cache_dir):
        """Test default TTL is correctly set."""
        cache_manager = CacheManager(cache_dir=temp_cache_dir, ttl_hours=12)

        # 12 hours = 12 * 3600 seconds
        assert cache_manager.ttl_seconds == 12 * 3600

    def test_cache_manager_uses_existing_directory(self, temp_cache_dir):
        """Test that existing cache directory is used without error."""
        # Directory already exists from fixture
        _cache_manager = CacheManager(cache_dir=temp_cache_dir, ttl_hours=24)

        assert Path(temp_cache_dir).exists()


class TestCacheSet:
    """Test cache storage functionality."""

    def test_set_stores_dataframe(self, cache_manager, sample_dataframe):
        """Test that DataFrame can be stored in cache."""
        cache_manager.set("AAPL", "1y", sample_dataframe, "prices")

        # Check that cache file was created
        cache_key = cache_manager._get_cache_key("AAPL", "1y", "prices")
        cache_path = cache_manager._get_cache_path(cache_key)

        assert cache_path.exists()

    def test_set_creates_parquet_file(self, cache_manager, sample_dataframe):
        """Test that cache uses parquet format."""
        cache_manager.set("MSFT", "6mo", sample_dataframe, "prices")

        cache_key = cache_manager._get_cache_key("MSFT", "6mo", "prices")
        cache_path = cache_manager._get_cache_path(cache_key)

        # File should have .parquet extension
        assert cache_path.suffix == ".parquet"

    def test_set_overwrites_existing_cache(self, cache_manager, sample_dataframe):
        """Test that setting cache twice overwrites the first entry."""
        # Set cache first time
        cache_manager.set("GOOGL", "1y", sample_dataframe, "prices")

        # Modify the dataframe
        modified_df = sample_dataframe.copy()
        modified_df["Close"] = modified_df["Close"] * 2

        # Set cache second time
        cache_manager.set("GOOGL", "1y", modified_df, "prices")

        # Retrieve and verify it's the modified version
        retrieved = cache_manager.get("GOOGL", "1y", "prices")
        assert retrieved is not None
        pd.testing.assert_series_equal(retrieved["Close"], modified_df["Close"])


class TestCacheGet:
    """Test cache retrieval functionality."""

    def test_get_returns_stored_dataframe(self, cache_manager, sample_dataframe):
        """Test that cached DataFrame can be retrieved."""
        cache_manager.set("AAPL", "1y", sample_dataframe, "prices")

        retrieved = cache_manager.get("AAPL", "1y", "prices")

        assert retrieved is not None
        pd.testing.assert_frame_equal(retrieved, sample_dataframe)

    def test_get_returns_none_for_missing_cache(self, cache_manager):
        """Test that get returns None when cache doesn't exist."""
        result = cache_manager.get("NONEXISTENT", "1y", "prices")

        assert result is None

    def test_get_returns_none_for_expired_cache(self, cache_manager, sample_dataframe):
        """Test that expired cache returns None."""
        # Create cache with very short TTL
        short_ttl_cache = CacheManager(
            cache_dir=cache_manager.cache_dir, ttl_hours=0.0001  # ~0.36 seconds
        )

        short_ttl_cache.set("AAPL", "1y", sample_dataframe, "prices")

        # Wait for cache to expire
        time.sleep(1)

        result = short_ttl_cache.get("AAPL", "1y", "prices")

        assert result is None

    def test_get_deletes_expired_cache_file(self, cache_manager, sample_dataframe):
        """Test that expired cache file is deleted."""
        short_ttl_cache = CacheManager(
            cache_dir=cache_manager.cache_dir, ttl_hours=0.0001
        )

        short_ttl_cache.set("AAPL", "1y", sample_dataframe, "prices")

        cache_key = short_ttl_cache._get_cache_key("AAPL", "1y", "prices")
        cache_path = short_ttl_cache._get_cache_path(cache_key)

        # Verify file exists
        assert cache_path.exists()

        # Wait for expiration
        time.sleep(1)

        # Get should return None and delete file
        short_ttl_cache.get("AAPL", "1y", "prices")

        assert not cache_path.exists()

    def test_get_handles_corrupted_cache(self, cache_manager, sample_dataframe):
        """Test handling of corrupted cache files."""
        # Store valid cache first
        cache_manager.set("AAPL", "1y", sample_dataframe, "prices")

        # Corrupt the cache file
        cache_key = cache_manager._get_cache_key("AAPL", "1y", "prices")
        cache_path = cache_manager._get_cache_path(cache_key)

        with open(cache_path, "wb") as f:
            f.write(b"corrupted data")

        # Get should return None and handle error gracefully
        result = cache_manager.get("AAPL", "1y", "prices")

        assert result is None
        # Corrupted file should be removed
        assert not cache_path.exists()


class TestCacheKey:
    """Test cache key generation."""

    def test_get_cache_key_format(self, cache_manager):
        """Test cache key format is consistent."""
        key = cache_manager._get_cache_key("AAPL", "1y", "prices")

        # Should be an MD5 hash (32 hexadecimal characters)
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

    def test_get_cache_key_is_unique(self, cache_manager):
        """Test that different inputs produce different keys."""
        key1 = cache_manager._get_cache_key("AAPL", "1y", "prices")
        key2 = cache_manager._get_cache_key("MSFT", "1y", "prices")
        key3 = cache_manager._get_cache_key("AAPL", "6mo", "prices")
        key4 = cache_manager._get_cache_key("AAPL", "1y", "fundamentals")

        assert key1 != key2
        assert key1 != key3
        assert key1 != key4

    def test_get_cache_key_is_deterministic(self, cache_manager):
        """Test that same inputs always produce same key."""
        key1 = cache_manager._get_cache_key("AAPL", "1y", "prices")
        key2 = cache_manager._get_cache_key("AAPL", "1y", "prices")

        assert key1 == key2


class TestCachePath:
    """Test cache path generation."""

    def test_get_cache_path_creates_path_object(self, cache_manager):
        """Test that cache path returns a Path object."""
        # Use a proper MD5 hash (32 hex chars) generated by _get_cache_key
        cache_key = cache_manager._get_cache_key("AAPL", "1y", "prices")
        cache_path = cache_manager._get_cache_path(cache_key)

        assert isinstance(cache_path, Path)

    def test_get_cache_path_has_parquet_extension(self, cache_manager):
        """Test that cache path has .parquet extension."""
        # Use a proper MD5 hash (32 hex chars) generated by _get_cache_key
        cache_key = cache_manager._get_cache_key("AAPL", "1y", "prices")
        cache_path = cache_manager._get_cache_path(cache_key)

        assert cache_path.suffix == ".parquet"

    def test_get_cache_path_is_in_cache_directory(self, cache_manager):
        """Test that cache path is within cache directory."""
        # Use a proper MD5 hash (32 hex chars) generated by _get_cache_key
        cache_key = cache_manager._get_cache_key("AAPL", "1y", "prices")
        cache_path = cache_manager._get_cache_path(cache_key)

        assert cache_path.parent == cache_manager.cache_dir


class TestCacheWithDifferentDataTypes:
    """Test cache with various DataFrame structures."""

    def test_cache_empty_dataframe(self, cache_manager):
        """Test caching an empty DataFrame."""
        empty_df = pd.DataFrame()

        cache_manager.set("EMPTY", "1y", empty_df, "test")
        retrieved = cache_manager.get("EMPTY", "1y", "test")

        assert retrieved is not None
        assert len(retrieved) == 0

    def test_cache_dataframe_with_various_dtypes(self, cache_manager):
        """Test caching DataFrame with mixed data types."""
        mixed_df = pd.DataFrame(
            {
                "string_col": ["a", "b", "c"],
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "bool_col": [True, False, True],
                "date_col": pd.date_range("2024-01-01", periods=3),
            }
        )

        cache_manager.set("MIXED", "1y", mixed_df, "test")
        retrieved = cache_manager.get("MIXED", "1y", "test")

        assert retrieved is not None
        pd.testing.assert_frame_equal(retrieved, mixed_df)

    def test_cache_large_dataframe(self, cache_manager):
        """Test caching a large DataFrame."""
        large_df = pd.DataFrame({"col" + str(i): range(10000) for i in range(10)})

        cache_manager.set("LARGE", "1y", large_df, "test")
        retrieved = cache_manager.get("LARGE", "1y", "test")

        assert retrieved is not None
        assert len(retrieved) == 10000
        pd.testing.assert_frame_equal(retrieved, large_df)


class TestCacheConcurrency:
    """Test cache behavior with concurrent access scenarios."""

    def test_multiple_different_caches(self, cache_manager, sample_dataframe):
        """Test storing multiple different cache entries."""
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        for ticker in tickers:
            df = sample_dataframe.copy()
            df["ticker"] = ticker
            cache_manager.set(ticker, "1y", df, "prices")

        # All should be retrievable
        for ticker in tickers:
            retrieved = cache_manager.get(ticker, "1y", "prices")
            assert retrieved is not None
            assert retrieved["ticker"].iloc[0] == ticker


class TestCacheCleanup:
    """Test cache cleanup and management."""

    def test_cache_file_removed_after_expiration(
        self, temp_cache_dir, sample_dataframe
    ):
        """Test that cache files are properly removed after expiration."""
        cache_manager = CacheManager(cache_dir=temp_cache_dir, ttl_hours=0.0001)

        cache_manager.set("AAPL", "1y", sample_dataframe, "prices")

        # Verify file exists
        cache_files_before = list(Path(temp_cache_dir).glob("*.parquet"))
        assert len(cache_files_before) == 1

        # Wait for expiration
        time.sleep(1)

        # Trigger cleanup by attempting to get
        cache_manager.get("AAPL", "1y", "prices")

        # Verify file is removed
        cache_files_after = list(Path(temp_cache_dir).glob("*.parquet"))
        assert len(cache_files_after) == 0


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_set_with_invalid_dataframe(self, cache_manager):
        """Test handling of invalid data types."""
        # Cache logs a warning and returns without raising exception
        cache_manager.set("AAPL", "1y", "not a dataframe", "prices")

        # Verify that no cache file was created
        cache_key = cache_manager._get_cache_key("AAPL", "1y", "prices")
        cache_path = cache_manager._get_cache_path(cache_key)
        assert not cache_path.exists()

    def test_get_handles_read_exception(self, cache_manager, sample_dataframe):
        """Test that get handles read exceptions gracefully."""
        # First set a valid cache
        cache_manager.set("AAPL", "1y", sample_dataframe, "prices")

        # Corrupt the cache file to trigger read error
        cache_key = cache_manager._get_cache_key("AAPL", "1y", "prices")
        cache_path = cache_manager._get_cache_path(cache_key)

        with open(cache_path, "wb") as f:
            f.write(b"corrupted parquet data")

        # Get should return None and handle exception
        result = cache_manager.get("AAPL", "1y", "prices")

        assert result is None
        # Corrupted file should be removed
        assert not cache_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
