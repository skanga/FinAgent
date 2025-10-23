"""
Utility functions and helpers.
"""

import time
import logging
from threading import Lock
from typing import List

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Thread-safe progress tracker for long-running operations."""

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
        self._lock = Lock()  # Thread-safe counter

    def update(self, item: str, success: bool = True) -> None:
        """
        Updates the progress in a thread-safe manner.

        Args:
            item (str): The item being processed.
            success (bool): Whether the operation was successful.
        """
        with self._lock:
            self.current += 1
            current = self.current
            elapsed = time.time() - self.start_time
            rate = current / elapsed if elapsed > 0 else 0
            eta = (self.total - current) / rate if rate > 0 else 0

            status = "✓" if success else "✗"
            print(
                f"  [{current}/{self.total}] {status} {item} | "
                f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s",
                flush=True,
            )

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

    normalized_period = user_period.strip().lower()
    valid_periods = [
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "ytd",
        "max",
    ]

    if normalized_period in valid_periods:
        return normalized_period

    # Natural language mappings
    mappings = {
        "quarter": "3mo",
        "month": "1mo",
        "year": "1y",
        "annual": "1y",
        "ytd": "ytd",
    }

    for key, value in mappings.items():
        if key in normalized_period:
            return value

    logger.warning(f"Unknown period '{user_period}', defaulting to 1y")
    return "1y"


def validate_ticker_symbol(ticker: str) -> str:
    """
    Validates and normalizes a single ticker symbol.

    This is the centralized ticker validation logic used across the application.

    Args:
        ticker (str): The ticker symbol to validate.

    Returns:
        str: The validated and normalized ticker symbol (uppercased and stripped).

    Raises:
        ValueError: If the ticker is invalid (empty, wrong length, invalid characters, or suspicious name).

    Examples:
        >>> validate_ticker_symbol("aapl")
        'AAPL'
        >>> validate_ticker_symbol("BRK.B")
        'BRK.B'
        >>> validate_ticker_symbol("")
        ValueError: Ticker cannot be empty
    """
    # Strip whitespace and convert to uppercase
    ticker = ticker.strip().upper()

    # Check for empty ticker
    if not ticker:
        raise ValueError("Ticker cannot be empty")

    # Check length (most ticker symbols are 1-10 characters)
    if len(ticker) < 1 or len(ticker) > 10:
        raise ValueError(
            f"Ticker '{ticker}' has invalid length (must be 1-10 characters)"
        )

    # Check for valid characters (alphanumeric, dots, hyphens)
    if not all(c.isalnum() or c in ".-" for c in ticker):
        raise ValueError(
            f"Invalid characters in ticker '{ticker}'. Only alphanumeric, dots, and hyphens allowed."
        )

    # Warn about suspicious tickers
    if ticker.lower() in ["test", "null", "none", "undefined"]:
        raise ValueError(f"Suspicious ticker name: '{ticker}'")

    return ticker


def validate_ticker_list(tickers: List[str]) -> List[str]:
    """
    Validates and normalizes a list of ticker symbols.

    This is the centralized ticker list validation logic used across the application.

    Args:
        tickers (List[str]): The list of ticker symbols to validate.

    Returns:
        List[str]: The validated and normalized list of ticker symbols.

    Raises:
        ValueError: If any ticker in the list is invalid.

    Examples:
        >>> validate_ticker_list(["aapl", "msft", "googl"])
        ['AAPL', 'MSFT', 'GOOGL']
        >>> validate_ticker_list([])
        ValueError: Ticker list cannot be empty
    """
    if not tickers:
        raise ValueError("Ticker list cannot be empty")

    validated = []
    for ticker in tickers:
        validated.append(validate_ticker_symbol(ticker))

    return validated
