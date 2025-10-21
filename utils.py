"""
Utility functions and helpers.
"""
import time
import logging

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Track and display progress for long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing") -> None:
        """
        Initializes the ProgressTracker.

        Args:
            total (int): The total number of items to track.
            description (str): A description of the operation being tracked.
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()

    def update(self, item: str, success: bool = True) -> None:
        """
        Updates the progress.

        Args:
            item (str): The item being processed.
            success (bool): Whether the operation was successful.
        """
        self.current += 1
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0

        status = "✓" if success else "✗"
        print(f"  [{self.current}/{self.total}] {status} {item} | "
              f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", flush=True)

    def complete(self) -> None:
        """
        Marks the progress as complete.
        """
        elapsed = time.time() - self.start_time
        print(f"✓ {self.description} complete in {elapsed:.1f}s")


def normalize_period(user_period: str) -> str:
    """
    Normalizes a user-provided time period to a yfinance-compatible period.

    Args:
        user_period (str): The user-provided time period.

    Returns:
        str: The normalized time period.
    """
    if not user_period:
        return "1y"
    
    p = user_period.strip().lower()
    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    
    if p in valid_periods:
        return p
    
    # Natural language mappings
    mappings = {
        "quarter": "3mo",
        "month": "1mo",
        "year": "1y",
        "annual": "1y",
        "ytd": "ytd"
    }
    
    for key, value in mappings.items():
        if key in p:
            return value
    
    logger.warning(f"Unknown period '{user_period}', defaulting to 1y")
    return "1y"
