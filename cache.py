"""
Caching system with TTL management.
"""

import time
import hashlib
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


class CacheManager:
    """Intelligent caching for API responses with TTL."""

    def __init__(self, cache_dir: str = "./.cache", ttl_hours: int = 24) -> None:
        """
        Initializes the CacheManager with a cache directory and a time-to-live (TTL).

        Args:
            cache_dir (str): The directory to store the cache files in.
            ttl_hours (int): The time-to-live for cache files in hours.
        """
        # Resolve and validate cache directory path
        self.cache_dir = Path(cache_dir).resolve()

        # Ensure cache directory is within current working directory or absolute safe path
        cwd = Path.cwd().resolve()
        try:
            # Check if cache_dir is relative to cwd
            self.cache_dir.relative_to(cwd)
        except ValueError:
            # If not relative to cwd, ensure it's not trying to escape system directories
            # Only allow absolute paths within user's home directory or temp directory
            home_dir = Path.home().resolve()
            import tempfile

            temp_dir = Path(tempfile.gettempdir()).resolve()

            if not (
                self._is_safe_subpath(self.cache_dir, home_dir)
                or self._is_safe_subpath(self.cache_dir, temp_dir)
            ):
                raise ValueError(
                    f"Cache directory must be within current working directory, "
                    f"home directory, or temp directory. Got: {cache_dir}"
                )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        logger.debug(f"Cache initialized: {self.cache_dir} (TTL: {ttl_hours}h)")

    def _is_safe_subpath(self, path: Path, parent: Path) -> bool:
        """Check if path is safely within parent directory."""
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    def _get_cache_key(self, ticker: str, period: str, data_type: str) -> str:
        """
        Generate secure cache key that cannot contain path separators.

        Uses MD5 hash to ensure the key is always a safe filename with no
        directory traversal characters (/, \, .., etc.).

        Args:
            ticker: Stock ticker symbol
            period: Time period for the data
            data_type: Type of data being cached

        Returns:
            32-character hexadecimal hash that is safe to use as filename
        """
        key_str = f"{ticker}_{period}_{data_type}"
        # MD5 hash produces only [0-9a-f] characters - no path separators possible
        return hashlib.md5(key_str.encode()).hexdigest()

    def _validate_cache_key(self, cache_key: str) -> None:
        """
        Validate cache key before using it in path construction.

        Ensures the key contains only safe characters and cannot be used
        for path traversal attacks.

        Args:
            cache_key: The cache key to validate

        Raises:
            ValueError: If cache key contains unsafe characters
        """
        # Cache key should only contain hexadecimal characters (from MD5)
        # This check is defensive - _get_cache_key always produces safe output
        if not cache_key:
            raise ValueError("Cache key cannot be empty")

        # Check length (MD5 hash is always 32 characters)
        if len(cache_key) != 32:
            raise ValueError(
                f"Invalid cache key length: {len(cache_key)} (expected 32)"
            )

        # Only allow hexadecimal characters [0-9a-f]
        if not all(c in "0123456789abcdef" for c in cache_key):
            raise ValueError(
                f"Cache key contains invalid characters. "
                f"Only hexadecimal [0-9a-f] allowed: {cache_key[:10]}..."
            )

        # Extra paranoia: ensure no path separators
        # (This should never trigger with proper MD5 hash, but defense in depth)
        forbidden_chars = {"/", "\\", "..", "~", ":"}
        if any(char in cache_key for char in forbidden_chars):
            raise ValueError(
                f"Cache key contains forbidden path characters: {cache_key}"
            )

    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get cache file path with comprehensive path traversal protection.

        Security measures:
        1. Validates cache key BEFORE path construction
        2. Uses only the key (no user input) in filename
        3. Verifies final path is within cache directory
        4. Never resolves symlinks that could escape cache dir

        Args:
            cache_key: Pre-validated MD5 hash from _get_cache_key()

        Returns:
            Safe path within cache directory

        Raises:
            ValueError: If path validation fails
        """
        # STEP 1: Validate cache key BEFORE using it
        self._validate_cache_key(cache_key)

        # STEP 2: Construct path using only validated key
        # Since cache_key is validated MD5 hash, this is safe
        filename = f"{cache_key}.parquet"
        cache_path = self.cache_dir / filename

        # STEP 3: Resolve to absolute path
        cache_path = cache_path.resolve()

        # STEP 4: Verify the resolved path is still within cache directory
        # This catches any edge cases with symlinks or unusual filesystem behavior
        try:
            cache_path.relative_to(self.cache_dir)
        except ValueError:
            # Path escaped cache directory
            raise ValueError(
                f"Security violation: Cache path outside cache directory. "
                f"Path: {cache_path}, Cache dir: {self.cache_dir}"
            )

        return cache_path

    def get(self, ticker: str, period: str, data_type: str = "prices") -> Optional[Any]:
        """
        Retrieves data from the cache if it exists and is not expired.

        Args:
            ticker (str): The ticker symbol.
            period (str): The time period for the data.
            data_type (str): The type of data to retrieve.

        Returns:
            Optional[Any]: The cached data, or None if it doesn't exist or is expired.
        """
        cache_key = self._get_cache_key(ticker, period, data_type)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        cache_age_seconds = time.time() - cache_path.stat().st_mtime
        if cache_age_seconds > self.ttl_seconds:
            logger.debug(f"Cache expired for {ticker} ({data_type})")
            cache_path.unlink()
            return None

        try:
            # Use parquet format for secure, efficient DataFrame storage
            data = pd.read_parquet(cache_path)
            logger.debug(
                f"Cache hit: {ticker} ({data_type}), age: {cache_age_seconds/3600:.1f}h"
            )
            return data
        except (OSError, pd.errors.ParserError, ValueError) as e:
            logger.debug(f"Cache read failed for {ticker}: {e}")
            # Remove corrupted cache file
            try:
                cache_path.unlink()
            except OSError:
                pass
            return None

    def set(
        self, ticker: str, period: str, data: Any, data_type: str = "prices"
    ) -> None:
        """
        Stores data in the cache.

        Args:
            ticker (str): The ticker symbol.
            period (str): The time period for the data.
            data (Any): The data to store.
            data_type (str): The type of data to store.
        """
        cache_key = self._get_cache_key(ticker, period, data_type)
        cache_path = self._get_cache_path(cache_key)

        try:
            # Validate that data is a pandas DataFrame
            if not isinstance(data, pd.DataFrame):
                logger.error(
                    f"Cache only supports pandas DataFrames, got {type(data).__name__}"
                )
                return

            # Use parquet format for secure, efficient DataFrame storage
            data.to_parquet(cache_path, compression="snappy", index=False)
            logger.debug(f"Cached {ticker} ({data_type})")
        except (OSError, ValueError, TypeError) as e:
            logger.debug(f"Cache write failed for {ticker}: {e}")

    def clear_expired(self) -> int:
        """
        Clears all expired cache entries.

        Returns:
            int: The number of cleared entries.
        """
        cleared = 0
        for cache_file in self.cache_dir.glob("*.parquet"):
            try:
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age > self.ttl_seconds:
                    cache_file.unlink()
                    cleared += 1
            except OSError as e:
                logger.debug(f"Failed to clear cache file {cache_file}: {e}")

        if cleared > 0:
            logger.debug(f"Cleared {cleared} expired cache entries")
        return cleared
