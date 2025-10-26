"""
Options data fetching with caching support.

This module provides cached fetching of options chain data from yfinance,
following the same patterns as fetcher.py for consistency.
"""

import logging
import pandas as pd
import yfinance as yf
from datetime import date, datetime
from typing import List, Optional
from cache import CacheManager
from utils import validate_ticker_symbol
from models_options import (
    OptionsContract,
    OptionsChain,
    OptionType,
    calculate_moneyness,
    calculate_intrinsic_value,
    is_itm,
    find_atm_strike,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


class OptionsDataFetcher:
    """Options data fetcher with intelligent caching."""

    def __init__(
        self,
        cache_manager: CacheManager,
        timeout: int = 30,
    ) -> None:
        """
        Initializes the OptionsDataFetcher with a cache manager.

        Args:
            cache_manager (CacheManager): The cache manager to use.
            timeout (int): The request timeout in seconds.
        """
        self.cache = cache_manager
        self.timeout = timeout

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((OSError, ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def fetch_available_expirations(self, ticker: str) -> List[date]:
        """
        Fetches available options expiration dates for a ticker.

        Automatically retries up to 3 times with exponential backoff for
        transient network errors.

        Args:
            ticker (str): The ticker symbol to fetch.

        Returns:
            List[date]: List of available expiration dates, sorted chronologically.

        Raises:
            ValueError: If ticker is invalid or has no options
            OSError: If network errors persist after retries
        """
        # Validate and normalize ticker using centralized validation
        ticker = validate_ticker_symbol(ticker)

        # Try cache first
        cached_data = self.cache.get(ticker, "options_expirations", "metadata")
        if cached_data is not None:
            logger.debug(f"Using cached expirations for {ticker}")
            return cached_data

        # Fetch fresh data
        logger.debug(f"Fetching options expirations for {ticker}")
        try:
            ticker_obj = yf.Ticker(ticker)
            expirations = ticker_obj.options

            if not expirations:
                raise ValueError(f"No options available for {ticker}")

            # Convert string dates to date objects
            expiration_dates = [
                datetime.strptime(exp, "%Y-%m-%d").date() for exp in expirations
            ]

            # Sort chronologically
            expiration_dates.sort()

            # Cache the result
            self.cache.set(ticker, "options_expirations", expiration_dates, "metadata")

            logger.debug(f"Found {len(expiration_dates)} expirations for {ticker}")
            return expiration_dates

        except ValueError:
            # Re-raise ValueError as-is (no options available)
            raise
        except OSError:
            # Re-raise OSError to trigger retry
            logger.error(f"Network error fetching options expirations for {ticker}")
            raise
        except (KeyError, AttributeError, TypeError, Exception) as e:
            # Handle yfinance API errors
            logger.error(f"Failed to fetch options expirations for {ticker}: {e}")
            raise ValueError(f"Failed to fetch options expirations for {ticker}: {str(e)}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((OSError, ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def fetch_options_chain(
        self, ticker: str, expiration: date, spot_price: Optional[float] = None
    ) -> OptionsChain:
        """
        Fetches the complete options chain for a specific expiration.

        Automatically retries up to 3 times with exponential backoff for
        transient network errors.

        Args:
            ticker (str): The ticker symbol.
            expiration (date): The expiration date.
            spot_price (Optional[float]): Current underlying price (fetched if not provided).

        Returns:
            OptionsChain: Complete options chain with calls and puts.

        Raises:
            ValueError: If ticker/expiration is invalid or has no data
            OSError: If network errors persist after retries
        """
        # Validate and normalize ticker using centralized validation
        ticker = validate_ticker_symbol(ticker)
        expiration_str = expiration.strftime("%Y-%m-%d")

        # Try cache first
        cached_data = self.cache.get(ticker, expiration_str, "options_chain")
        if cached_data is not None:
            logger.debug(f"Using cached options chain for {ticker} exp {expiration_str}")
            return cached_data

        # Fetch fresh data
        logger.debug(f"Fetching options chain for {ticker} exp {expiration_str}")
        try:
            ticker_obj = yf.Ticker(ticker)

            # Get current price if not provided
            if spot_price is None:
                info = ticker_obj.info
                spot_price = info.get("currentPrice") or info.get("regularMarketPrice")
                if spot_price is None:
                    # Try to get from history
                    hist = ticker_obj.history(period="1d")
                    if not hist.empty:
                        spot_price = float(hist["Close"].iloc[-1])
                    else:
                        raise ValueError(f"Could not determine current price for {ticker}")

            # Fetch options chain
            opt = ticker_obj.option_chain(expiration_str)

            # Parse calls
            calls = self._parse_contracts(
                opt.calls, ticker, expiration, OptionType.CALL, spot_price
            )

            # Parse puts
            puts = self._parse_contracts(
                opt.puts, ticker, expiration, OptionType.PUT, spot_price
            )

            # Calculate aggregate metrics
            total_call_volume = sum(c.volume or 0 for c in calls)
            total_put_volume = sum(p.volume or 0 for p in puts)
            total_call_oi = sum(c.open_interest or 0 for c in calls)
            total_put_oi = sum(p.open_interest or 0 for p in puts)

            put_call_ratio_volume = (
                total_put_volume / total_call_volume
                if total_call_volume > 0
                else None
            )
            put_call_ratio_oi = (
                total_put_oi / total_call_oi if total_call_oi > 0 else None
            )

            # Find ATM IV
            call_strikes = [c.strike for c in calls]
            put_strikes = [p.strike for p in puts]
            atm_strike = find_atm_strike(spot_price, call_strikes + put_strikes)

            atm_call_iv = None
            atm_put_iv = None
            if atm_strike:
                atm_calls = [c for c in calls if c.strike == atm_strike]
                atm_puts = [p for p in puts if p.strike == atm_strike]
                if atm_calls and atm_calls[0].implied_volatility:
                    atm_call_iv = atm_calls[0].implied_volatility
                if atm_puts and atm_puts[0].implied_volatility:
                    atm_put_iv = atm_puts[0].implied_volatility

            # Create chain object
            chain = OptionsChain(
                ticker=ticker,
                expiration=expiration,
                underlying_price=spot_price,
                calls=calls,
                puts=puts,
                total_call_volume=total_call_volume,
                total_put_volume=total_put_volume,
                total_call_oi=total_call_oi,
                total_put_oi=total_put_oi,
                put_call_ratio_volume=put_call_ratio_volume,
                put_call_ratio_oi=put_call_ratio_oi,
                atm_call_iv=atm_call_iv,
                atm_put_iv=atm_put_iv,
            )

            # Cache the result
            self.cache.set(ticker, expiration_str, chain, "options_chain")

            logger.debug(
                f"Fetched chain for {ticker} exp {expiration_str}: "
                f"{len(calls)} calls, {len(puts)} puts"
            )
            return chain

        except ValueError:
            # Re-raise ValueError as-is
            raise
        except OSError:
            # Re-raise OSError to trigger retry
            logger.error(f"Network error fetching options chain for {ticker}")
            raise
        except (KeyError, AttributeError, TypeError, Exception) as e:
            # Handle yfinance API errors
            logger.error(f"Failed to fetch options chain for {ticker} exp {expiration_str}: {e}")
            raise ValueError(
                f"Failed to fetch options chain for {ticker} exp {expiration_str}: {str(e)}"
            ) from e

    def _parse_contracts(
        self,
        df: pd.DataFrame,
        ticker: str,
        expiration: date,
        option_type: OptionType,
        spot_price: float,
    ) -> List[OptionsContract]:
        """
        Parse options contracts from yfinance DataFrame.

        Uses itertuples() for faster iteration compared to iterrows().

        Args:
            df: yfinance options DataFrame
            ticker: Ticker symbol
            expiration: Expiration date
            option_type: Call or Put
            spot_price: Current underlying price

        Returns:
            List of OptionsContract objects
        """
        contracts = []

        # Use itertuples() for 100x faster iteration than iterrows()
        for row in df.itertuples(index=False):
            try:
                strike = float(row.strike)

                # Calculate derived metrics
                moneyness = calculate_moneyness(spot_price, strike, option_type)
                intrinsic_value = calculate_intrinsic_value(spot_price, strike, option_type)
                in_the_money = is_itm(spot_price, strike, option_type)

                # Extract last price
                last_price = None
                if hasattr(row, "lastPrice") and pd.notna(row.lastPrice):
                    last_price = float(row.lastPrice)

                # Extract bid/ask
                bid = float(row.bid) if hasattr(row, "bid") and pd.notna(row.bid) else None
                ask = float(row.ask) if hasattr(row, "ask") and pd.notna(row.ask) else None

                # Calculate extrinsic value
                extrinsic_value = None
                if last_price is not None:
                    extrinsic_value = last_price - intrinsic_value

                # Extract volume and OI
                volume = int(row.volume) if hasattr(row, "volume") and pd.notna(row.volume) else None
                open_interest = (
                    int(row.openInterest)
                    if hasattr(row, "openInterest") and pd.notna(row.openInterest)
                    else None
                )

                # Extract IV
                implied_volatility = (
                    float(row.impliedVolatility)
                    if hasattr(row, "impliedVolatility") and pd.notna(row.impliedVolatility)
                    else None
                )

                # Extract contract symbol
                contract_symbol = (
                    str(row.contractSymbol)
                    if hasattr(row, "contractSymbol") and pd.notna(row.contractSymbol)
                    else None
                )

                # Extract last trade date
                last_trade_date = None
                if hasattr(row, "lastTradeDate") and pd.notna(row.lastTradeDate):
                    try:
                        last_trade_date = pd.to_datetime(row.lastTradeDate)
                    except (ValueError, TypeError):
                        pass

                # Create contract
                contract = OptionsContract(
                    ticker=ticker,
                    strike=strike,
                    expiration=expiration,
                    option_type=option_type,
                    last_price=last_price,
                    bid=bid,
                    ask=ask,
                    volume=volume,
                    open_interest=open_interest,
                    implied_volatility=implied_volatility,
                    in_the_money=in_the_money,
                    intrinsic_value=intrinsic_value,
                    extrinsic_value=extrinsic_value,
                    moneyness=moneyness,
                    contract_symbol=contract_symbol,
                    last_trade_date=last_trade_date,
                )

                contracts.append(contract)

            except (ValueError, KeyError, TypeError) as e:
                logger.debug(f"Skipping contract due to parsing error: {e}")
                continue

        return contracts

    def fetch_multiple_expirations(
        self, ticker: str, num_expirations: int = 3
    ) -> List[OptionsChain]:
        """
        Fetch options chains for multiple expirations.

        Args:
            ticker: Ticker symbol
            num_expirations: Number of expirations to fetch (chronologically)

        Returns:
            List of OptionsChain objects

        Raises:
            ValueError: If ticker has no options or insufficient expirations
        """
        # Validate and normalize ticker using centralized validation
        ticker = validate_ticker_symbol(ticker)

        # Get available expirations
        expirations = self.fetch_available_expirations(ticker)

        if len(expirations) < num_expirations:
            logger.warning(
                f"{ticker} has only {len(expirations)} expirations, "
                f"requested {num_expirations}"
            )
            num_expirations = len(expirations)

        if num_expirations == 0:
            raise ValueError(f"No expirations available for {ticker}")

        # Fetch first N expirations
        chains = []
        spot_price = None  # Fetch once and reuse

        for exp in expirations[:num_expirations]:
            try:
                chain = self.fetch_options_chain(ticker, exp, spot_price)
                chains.append(chain)

                # Cache spot price for subsequent calls
                if spot_price is None:
                    spot_price = chain.underlying_price

            except (ValueError, OSError) as e:
                logger.error(f"Failed to fetch chain for {ticker} exp {exp}: {e}")
                continue

        if not chains:
            raise ValueError(f"Failed to fetch any options chains for {ticker}")

        logger.info(f"Fetched {len(chains)} options chains for {ticker}")
        return chains

    def get_current_price(self, ticker: str) -> float:
        """
        Get current price for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Current price

        Raises:
            ValueError: If price cannot be determined
        """
        # Validate and normalize ticker using centralized validation
        ticker = validate_ticker_symbol(ticker)

        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            price = info.get("currentPrice") or info.get("regularMarketPrice")

            if price is None:
                # Try history
                hist = ticker_obj.history(period="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
                else:
                    raise ValueError(f"Could not determine price for {ticker}")

            return float(price)

        except (KeyError, AttributeError, TypeError, OSError) as e:
            logger.error(f"Failed to get price for {ticker}: {e}")
            raise ValueError(f"Failed to get price for {ticker}: {str(e)}") from e
