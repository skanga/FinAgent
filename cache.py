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

            if not (self._is_safe_subpath(self.cache_dir, home_dir) or
                    self._is_safe_subpath(self.cache_dir, temp_dir)):
                raise ValueError(
                    f"Cache directory must be within current working directory, "
                    f"home directory, or temp directory. Got: {cache_dir}"
                )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        logger.info(f"Cache initialized: {self.cache_dir} (TTL: {ttl_hours}h)")

    def _is_safe_subpath(self, path: Path, parent: Path) -> bool:
        """Check if path is safely within parent directory."""
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False
    
    def _get_cache_key(self, ticker: str, period: str, data_type: str) -> str:
        """Generate cache key."""
        key_str = f"{ticker}_{period}_{data_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path with path traversal protection."""
        cache_path = (self.cache_dir / f"{cache_key}.parquet").resolve()

        # Ensure the resolved path is still within cache directory (prevent traversal)
        if not self._is_safe_subpath(cache_path, self.cache_dir):
            raise ValueError(f"Invalid cache path: attempted path traversal")

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

        cache_age = time.time() - cache_path.stat().st_mtime
        if cache_age > self.ttl_seconds:
            logger.info(f"Cache expired for {ticker} ({data_type})")
            cache_path.unlink()
            return None

        try:
            # Use parquet format for secure, efficient DataFrame storage
            data = pd.read_parquet(cache_path)
            logger.info(f"Cache hit: {ticker} ({data_type}), age: {cache_age/3600:.1f}h")
            return data
        except (OSError, pd.errors.ParserError, ValueError) as e:
            logger.warning(f"Cache read failed for {ticker}: {e}")
            # Remove corrupted cache file
            try:
                cache_path.unlink()
            except OSError:
                pass
            return None
    
    def set(self, ticker: str, period: str, data: Any, data_type: str = "prices") -> None:
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
                logger.warning(f"Cache only supports pandas DataFrames, got {type(data).__name__}")
                return

            # Use parquet format for secure, efficient DataFrame storage
            data.to_parquet(cache_path, compression='snappy', index=False)
            logger.info(f"Cached {ticker} ({data_type})")
        except (OSError, ValueError, TypeError) as e:
            logger.warning(f"Cache write failed for {ticker}: {e}")
    
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
                logger.warning(f"Failed to clear cache file {cache_file}: {e}")

        if cleared > 0:
            logger.info(f"Cleared {cleared} expired cache entries")
        return cleared