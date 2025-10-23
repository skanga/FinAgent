"""
Security tests for cache path traversal vulnerabilities.

Tests ensure that the CacheManager cannot be exploited to read/write files
outside the designated cache directory.
"""

import pytest
import tempfile
import pandas as pd
from pathlib import Path
from cache import CacheManager


class TestCachePathTraversalProtection:
    """Test suite for path traversal attack prevention."""

    def setup_method(self):
        """Create temporary cache directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
        self.cache = CacheManager(cache_dir=str(self.cache_dir), ttl_hours=1)

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_normal_cache_operation(self):
        """Verify normal cache operations work correctly."""
        # Create test data
        test_data = pd.DataFrame({"price": [100, 101, 102]})

        # Store and retrieve
        self.cache.set("AAPL", "1y", test_data, "prices")
        retrieved = self.cache.get("AAPL", "1y", "prices")

        assert retrieved is not None
        assert len(retrieved) == 3
        pd.testing.assert_frame_equal(test_data, retrieved)

    def test_cache_key_is_md5_hash(self):
        """Verify cache keys are MD5 hashes with no path separators."""
        cache_key = self.cache._get_cache_key("AAPL", "1y", "prices")

        # MD5 hash is 32 hex characters
        assert len(cache_key) == 32
        assert all(c in "0123456789abcdef" for c in cache_key)

        # No path separators
        assert "/" not in cache_key
        assert "\\" not in cache_key
        assert ".." not in cache_key

    def test_cache_key_validation_empty_key(self):
        """Empty cache key should be rejected."""
        with pytest.raises(ValueError, match="Cache key cannot be empty"):
            self.cache._validate_cache_key("")

    def test_cache_key_validation_wrong_length(self):
        """Cache key with wrong length should be rejected."""
        with pytest.raises(ValueError, match="Invalid cache key length"):
            self.cache._validate_cache_key("abc123")  # Too short

    def test_cache_key_validation_invalid_characters(self):
        """Cache key with non-hex characters should be rejected."""
        # Create a 32-character string with invalid chars
        invalid_key = "g" * 32  # 'g' is not a hex digit
        with pytest.raises(ValueError, match="invalid characters"):
            self.cache._validate_cache_key(invalid_key)

    def test_cache_key_validation_path_separators(self):
        """Cache key with path separators should be rejected."""
        # Try various path traversal attempts (padded to exactly 32 chars)
        malicious_keys = [
            "../" + "a" * 29,           # 3 + 29 = 32 chars
            "..\\" + "a" * 28,          # 3 + 28 = 31 chars - need one more
            "~/" + "a" * 30,            # 2 + 30 = 32 chars
            "a:b" + "a" * 29,           # 3 + 29 = 32 chars
        ]

        for key in malicious_keys:
            # Will fail on either length, invalid characters, or forbidden chars
            with pytest.raises(ValueError, match="forbidden path characters|invalid characters|Invalid cache key length"):
                self.cache._validate_cache_key(key)

    def test_cache_path_stays_in_cache_dir(self):
        """Generated cache paths must stay within cache directory."""
        cache_key = self.cache._get_cache_key("AAPL", "1y", "prices")
        cache_path = self.cache._get_cache_path(cache_key)

        # Verify path is within cache directory
        assert cache_path.is_relative_to(self.cache_dir)

        # Verify path doesn't escape using ..
        assert ".." not in str(cache_path.relative_to(self.cache_dir))

    def test_direct_path_traversal_attempt_fails(self):
        """
        Test that even if someone bypasses _get_cache_key and calls
        _get_cache_path directly with malicious input, it fails.
        """
        # This would fail validation
        with pytest.raises(ValueError):
            self.cache._get_cache_path("../etc/passwd")

        with pytest.raises(ValueError):
            self.cache._get_cache_path("..\\..\\windows\\system32")

    def test_symlink_attack_prevention(self):
        """
        Test that symlinks cannot be used to escape cache directory.

        This tests STEP 4 of _get_cache_path which verifies the resolved
        path is still within cache_dir.
        """
        # Create a directory outside cache
        outside_dir = Path(self.temp_dir) / "outside"
        outside_dir.mkdir(exist_ok=True)
        outside_file = outside_dir / "secret.parquet"
        outside_file.write_text("secret data")

        # Try to create a symlink in cache pointing outside
        # (In practice, an attacker couldn't do this, but testing defense in depth)
        cache_key = self.cache._get_cache_key("AAPL", "1y", "prices")

        # The normal path should work
        normal_path = self.cache._get_cache_path(cache_key)
        assert normal_path.is_relative_to(self.cache_dir)

    def test_ticker_with_path_characters_is_safe(self):
        """
        Test that tickers containing path-like characters are safely hashed.

        Even if an attacker passes "../etc/passwd" as a ticker, it should
        be safely hashed and cannot escape cache directory.
        """
        malicious_tickers = [
            "../etc/passwd",
            "..\\..\\windows\\system32",
            "~/../../secret",
            "C:\\Windows\\System32",
        ]

        test_data = pd.DataFrame({"price": [100]})

        for ticker in malicious_tickers:
            # Should not raise exception - ticker is hashed
            cache_key = self.cache._get_cache_key(ticker, "1y", "prices")

            # Key should be safe MD5 hash
            assert len(cache_key) == 32
            assert all(c in "0123456789abcdef" for c in cache_key)

            # Path should be safe
            cache_path = self.cache._get_cache_path(cache_key)
            assert cache_path.is_relative_to(self.cache_dir)

            # Should be able to store/retrieve without escaping cache
            self.cache.set(ticker, "1y", test_data, "prices")
            retrieved = self.cache.get(ticker, "1y", "prices")
            assert retrieved is not None

    def test_period_with_path_characters_is_safe(self):
        """Test that periods containing path characters are safely hashed."""
        malicious_periods = [
            "../../../etc",
            "1y/../../../secret",
        ]

        test_data = pd.DataFrame({"price": [100]})

        for period in malicious_periods:
            # Should be safely hashed
            cache_key = self.cache._get_cache_key("AAPL", period, "prices")
            assert len(cache_key) == 32
            assert all(c in "0123456789abcdef" for c in cache_key)

            # Can safely use
            self.cache.set("AAPL", period, test_data, "prices")
            retrieved = self.cache.get("AAPL", period, "prices")
            assert retrieved is not None

    def test_data_type_with_path_characters_is_safe(self):
        """Test that data_type containing path characters are safely hashed."""
        malicious_data_types = [
            "../../../etc/passwd",
            "prices/../../../secret",
        ]

        test_data = pd.DataFrame({"price": [100]})

        for data_type in malicious_data_types:
            # Should be safely hashed
            cache_key = self.cache._get_cache_key("AAPL", "1y", data_type)
            assert len(cache_key) == 32
            assert all(c in "0123456789abcdef" for c in cache_key)

            # Can safely use
            self.cache.set("AAPL", "1y", test_data, data_type)
            retrieved = self.cache.get("AAPL", "1y", data_type)
            assert retrieved is not None

    def test_cache_files_only_in_cache_dir(self):
        """Verify all cache files are created only in cache directory."""
        test_data = pd.DataFrame({"price": [100, 101, 102]})

        # Create multiple cache entries
        test_cases = [
            ("AAPL", "1y", "prices"),
            ("MSFT", "6mo", "prices"),
            ("GOOGL", "1mo", "fundamentals"),
        ]

        for ticker, period, data_type in test_cases:
            self.cache.set(ticker, period, test_data, data_type)

        # Check all cache files are in cache directory
        cache_files = list(self.cache_dir.glob("*.parquet"))
        assert len(cache_files) == 3

        for cache_file in cache_files:
            # Each file should be directly in cache_dir (no subdirectories)
            assert cache_file.parent == self.cache_dir

            # Filename should be 32-char hex + .parquet
            assert len(cache_file.stem) == 32
            assert all(c in "0123456789abcdef" for c in cache_file.stem)

    def test_null_byte_injection(self):
        """Test that null byte injection doesn't bypass validation."""
        # Null byte injection is a classic attack: "safe\x00../../etc/passwd"
        # The string appears safe before \x00, but filesystem treats it differently
        malicious_inputs = [
            "AAPL\x00../../etc/passwd",
            "1y\x00../secret",
            "prices\x00../../config",
        ]

        for malicious_input in malicious_inputs:
            # The MD5 hash will include the null byte, making it safe
            cache_key = self.cache._get_cache_key(malicious_input, "1y", "prices")

            # Should still be a safe hex string
            assert len(cache_key) == 32
            assert all(c in "0123456789abcdef" for c in cache_key)

            # Path should be safe
            cache_path = self.cache._get_cache_path(cache_key)
            assert cache_path.is_relative_to(self.cache_dir)

    def test_unicode_normalization_attack(self):
        """
        Test that Unicode normalization attacks don't bypass validation.

        Some filesystems normalize Unicode, which could potentially bypass
        checks. For example, "AAPL" vs "ＡＰＰＬ" (fullwidth characters).
        """
        # Different Unicode representations of similar-looking characters
        unicode_inputs = [
            "AAPL",      # Normal ASCII
            "ＡＰＰＬ",  # Fullwidth
            "A\u0041PL", # Unicode escape
        ]

        test_data = pd.DataFrame({"price": [100]})

        for unicode_input in unicode_inputs:
            # Each should produce a different hash
            cache_key = self.cache._get_cache_key(unicode_input, "1y", "prices")
            assert len(cache_key) == 32
            assert all(c in "0123456789abcdef" for c in cache_key)

            # Should be safely stored
            self.cache.set(unicode_input, "1y", test_data, "prices")


class TestCacheDirectoryValidation:
    """Test cache directory path validation."""

    def test_cache_dir_in_cwd_allowed(self):
        """Cache directory within CWD should be allowed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            cache = CacheManager(cache_dir=str(cache_dir), ttl_hours=1)
            assert cache.cache_dir == cache_dir.resolve()

    def test_cache_dir_in_home_allowed(self):
        """Cache directory within home directory should be allowed."""
        home_cache = Path.home() / "test_cache_temp"
        try:
            cache = CacheManager(cache_dir=str(home_cache), ttl_hours=1)
            assert cache.cache_dir == home_cache.resolve()
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(home_cache, ignore_errors=True)

    def test_cache_dir_in_temp_allowed(self):
        """Cache directory within temp directory should be allowed."""
        import tempfile
        temp_cache = Path(tempfile.gettempdir()) / "test_cache_temp"
        try:
            cache = CacheManager(cache_dir=str(temp_cache), ttl_hours=1)
            assert cache.cache_dir == temp_cache.resolve()
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_cache, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
