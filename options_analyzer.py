"""
Options analysis engine for Greeks, IV, and strategy detection.

This module provides comprehensive options analysis including:
- Black-Scholes-Merton Greeks calculation
- Implied Volatility solving
- Strategy detection and P&L analysis
- Risk metrics and recommendations
"""

import logging
import numpy as np
from typing import List, Optional
from scipy.stats import norm
from scipy.optimize import brentq, newton
from models_options import (
    OptionsContract,
    OptionsChain,
    GreeksData,
    OptionsStrategy,
    StrategyType,
    StrategyLeg,
    PnLScenario,
    IVAnalysis,
    OptionType,
    find_atm_strike,
)
from constants import OptionsAnalysisParameters

logger = logging.getLogger(__name__)


class OptionsAnalyzer:
    """Comprehensive options analysis engine."""

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """
        Initialize the OptionsAnalyzer.

        Args:
            risk_free_rate: Annual risk-free rate as decimal (default 2%)
        """
        self.risk_free_rate = risk_free_rate

    # ========================================================================
    # BLACK-SCHOLES-MERTON MODEL
    # ========================================================================

    def _d1(
        self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
    ) -> float:
        """
        Calculate d1 parameter for Black-Scholes formula.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Volatility (annualized)
            q: Dividend yield (default 0)

        Returns:
            d1 value
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (
            sigma * np.sqrt(T)
        )

    def _d2(
        self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
    ) -> float:
        """
        Calculate d2 parameter for Black-Scholes formula.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Volatility (annualized)
            q: Dividend yield (default 0)

        Returns:
            d2 value
        """
        return self._d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)

    def calculate_bsm_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> float:
        """
        Calculate option price using Black-Scholes-Merton formula.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Volatility (annualized)
            option_type: CALL or PUT
            q: Dividend yield (default 0)

        Returns:
            Theoretical option price
        """
        if T <= 0:
            # Expired option, return intrinsic value
            if option_type == OptionType.CALL:
                return max(0.0, S - K)
            else:
                return max(0.0, K - S)

        d1 = self._d1(S, K, T, r, sigma, q)
        d2 = self._d2(S, K, T, r, sigma, q)

        if option_type == OptionType.CALL:
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # PUT
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        return max(0.0, price)

    # ========================================================================
    # GREEKS CALCULATION
    # ========================================================================

    def calculate_greeks(
        self,
        contract: OptionsContract,
        spot: float,
        rate: Optional[float] = None,
        dividend_yield: float = 0.0,
    ) -> GreeksData:
        """
        Calculate all Greeks for an options contract using Black-Scholes-Merton.

        Args:
            contract: OptionsContract to analyze
            spot: Current underlying price
            rate: Risk-free rate (uses default if None)
            dividend_yield: Annual dividend yield as decimal

        Returns:
            GreeksData with all Greeks populated
        """
        if rate is None:
            rate = self.risk_free_rate

        # Time to expiration in years
        T = contract.days_to_expiration / 365.0

        if T <= 0:
            # Expired option
            return GreeksData(
                delta=1.0 if contract.in_the_money else 0.0,
                gamma=0.0,
                theta=0.0,
                vega=0.0,
                rho=0.0,
            )

        # Clamp T to minimum to avoid division by zero in Gamma and Theta calculations
        # When T is very small (< 1 day), sqrt(T) approaches 0, causing numerical instability
        T = max(T, OptionsAnalysisParameters.MIN_TIME_TO_EXPIRATION)

        # Use contract's IV if available, otherwise use a default
        sigma = contract.implied_volatility if contract.implied_volatility else 0.30

        K = contract.strike
        S = spot
        q = dividend_yield

        # Calculate d1 and d2
        d1 = self._d1(S, K, T, rate, sigma, q)
        d2 = self._d2(S, K, T, rate, sigma, q)

        # Delta
        if contract.option_type == OptionType.CALL:
            delta = np.exp(-q * T) * norm.cdf(d1)
        else:  # PUT
            delta = np.exp(-q * T) * (norm.cdf(d1) - 1)

        # Gamma (same for calls and puts)
        gamma = (
            np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            if sigma > 0 and T > 0
            else 0.0
        )

        # Vega (same for calls and puts) - per 1% change in volatility
        vega = (
            S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) * 0.01
            if T > 0
            else 0.0
        )

        # Theta (different for calls and puts) - per day
        common_theta_term = (
            -np.exp(-q * T) * (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            if T > 0
            else 0.0
        )

        if contract.option_type == OptionType.CALL:
            theta = (
                common_theta_term
                - rate * K * np.exp(-rate * T) * norm.cdf(d2)
                + q * S * np.exp(-q * T) * norm.cdf(d1)
            ) / 365.0  # Convert to per-day
        else:  # PUT
            theta = (
                common_theta_term
                + rate * K * np.exp(-rate * T) * norm.cdf(-d2)
                - q * S * np.exp(-q * T) * norm.cdf(-d1)
            ) / 365.0  # Convert to per-day

        # Rho (different for calls and puts) - per 1% change in interest rate
        if contract.option_type == OptionType.CALL:
            rho = K * T * np.exp(-rate * T) * norm.cdf(d2) * 0.01
        else:  # PUT
            rho = -K * T * np.exp(-rate * T) * norm.cdf(-d2) * 0.01

        # Delta and Gamma in dollars (assuming 100 shares per contract)
        delta_dollars = delta * 100
        gamma_dollars = gamma * 100

        return GreeksData(
            delta=float(delta),
            gamma=float(gamma),
            theta=float(theta),
            vega=float(vega),
            rho=float(rho),
            delta_dollars=float(delta_dollars),
            gamma_dollars=float(gamma_dollars),
        )

    # ========================================================================
    # IMPLIED VOLATILITY CALCULATION
    # ========================================================================

    def calculate_implied_volatility(
        self,
        contract: OptionsContract,
        spot: float,
        market_price: float,
        rate: Optional[float] = None,
        dividend_yield: float = 0.0,
    ) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method with fallback to Brent.

        Args:
            contract: OptionsContract to analyze
            spot: Current underlying price
            market_price: Observed market price of the option
            rate: Risk-free rate (uses default if None)
            dividend_yield: Annual dividend yield as decimal

        Returns:
            Implied volatility as decimal, or None if calculation fails
        """
        if rate is None:
            rate = self.risk_free_rate

        T = contract.days_to_expiration / 365.0
        if T <= 0:
            return None

        K = contract.strike
        S = spot

        # Intrinsic value check
        intrinsic = contract.intrinsic_value or 0.0
        if market_price <= intrinsic * 1.001:  # Within 0.1% of intrinsic
            return OptionsAnalysisParameters.IV_MIN

        # Define objective function
        def objective(sigma: float) -> float:
            try:
                theoretical = self.calculate_bsm_price(
                    S, K, T, rate, sigma, contract.option_type, dividend_yield
                )
                return theoretical - market_price
            except (ValueError, ZeroDivisionError, OverflowError):
                return float("inf")

        # Try Newton-Raphson first (faster when it converges)
        try:
            # Use Vega as derivative
            def vega_func(sigma: float) -> float:
                d1 = self._d1(S, K, T, rate, sigma, dividend_yield)
                return S * np.exp(-dividend_yield * T) * norm.pdf(d1) * np.sqrt(T)

            # Initial guess - use historical volatility as starting point
            initial_guess = 0.30

            iv = newton(
                objective,
                x0=initial_guess,
                fprime=vega_func,
                maxiter=OptionsAnalysisParameters.IV_SOLVER_MAX_ITERATIONS,
                tol=OptionsAnalysisParameters.IV_SOLVER_TOLERANCE,
            )

            # Validate result
            if (
                OptionsAnalysisParameters.IV_MIN
                <= iv
                <= OptionsAnalysisParameters.IV_MAX
            ):
                return float(iv)

        except (RuntimeError, ValueError, ZeroDivisionError):
            # Newton-Raphson failed, try Brent's method
            logger.debug("Newton-Raphson failed, falling back to Brent's method")

        # Fallback to Brent's method (more robust)
        try:
            iv = brentq(
                objective,
                OptionsAnalysisParameters.IV_MIN,
                OptionsAnalysisParameters.IV_MAX,
                maxiter=OptionsAnalysisParameters.IV_SOLVER_MAX_ITERATIONS,
                xtol=OptionsAnalysisParameters.IV_SOLVER_TOLERANCE,
            )
            return float(iv)

        except (ValueError, RuntimeError):
            logger.debug(
                f"IV calculation failed for {contract.ticker} "
                f"{contract.strike} {contract.option_type.value}"
            )
            return None

    def enrich_chain_with_greeks(
        self,
        chain: OptionsChain,
        calculate_iv: bool = False,
        dividend_yield: float = 0.0,
    ) -> OptionsChain:
        """
        Calculate Greeks for all contracts in a chain.

        Args:
            chain: OptionsChain to enrich
            calculate_iv: If True, recalculate IV from market prices
            dividend_yield: Annual dividend yield as decimal

        Returns:
            OptionsChain with Greeks populated
        """
        spot = chain.underlying_price

        # Process calls
        for contract in chain.calls:
            # Recalculate IV if requested and market price available
            if calculate_iv and contract.mid_price:
                iv = self.calculate_implied_volatility(
                    contract, spot, contract.mid_price, dividend_yield=dividend_yield
                )
                if iv is not None:
                    contract.implied_volatility = iv

            # Calculate Greeks
            contract.greeks = self.calculate_greeks(
                contract, spot, dividend_yield=dividend_yield
            )

        # Process puts
        for contract in chain.puts:
            # Recalculate IV if requested and market price available
            if calculate_iv and contract.mid_price:
                iv = self.calculate_implied_volatility(
                    contract, spot, contract.mid_price, dividend_yield=dividend_yield
                )
                if iv is not None:
                    contract.implied_volatility = iv

            # Calculate Greeks
            contract.greeks = self.calculate_greeks(
                contract, spot, dividend_yield=dividend_yield
            )

        return chain

    # ========================================================================
    # IV ANALYSIS
    # ========================================================================

    def analyze_implied_volatility(
        self,
        chains: List[OptionsChain],
        historical_volatility: float,
        ticker: str,
    ) -> IVAnalysis:
        """
        Analyze implied volatility patterns across strikes and expirations.

        Args:
            chains: List of options chains
            historical_volatility: Realized historical volatility from stock analysis
            ticker: Ticker symbol

        Returns:
            IVAnalysis with skew and term structure
        """
        if not chains:
            return IVAnalysis(
                ticker=ticker,
                current_iv=0.0,
                historical_volatility=historical_volatility,
            )

        # Get ATM IV from nearest expiration
        nearest_chain = chains[0]
        current_iv = nearest_chain.atm_call_iv or nearest_chain.atm_put_iv or 0.0

        # Build IV term structure (ATM IV by expiration)
        iv_term_structure = {}
        for chain in chains:
            atm_iv = chain.atm_call_iv or chain.atm_put_iv
            if atm_iv:
                iv_term_structure[chain.expiration] = atm_iv

        # Calculate skew for nearest expiration
        call_skew = None
        put_skew = None

        if nearest_chain.calls and nearest_chain.puts:
            spot = nearest_chain.underlying_price

            # Find OTM calls (strike > spot)
            otm_calls = [c for c in nearest_chain.calls if c.strike > spot * 1.05]
            if otm_calls and nearest_chain.atm_call_iv:
                otm_call_ivs = [c.implied_volatility for c in otm_calls if c.implied_volatility]
                if otm_call_ivs:
                    call_skew = np.mean(otm_call_ivs) - nearest_chain.atm_call_iv

            # Find OTM puts (strike < spot)
            otm_puts = [p for p in nearest_chain.puts if p.strike < spot * 0.95]
            if otm_puts and nearest_chain.atm_put_iv:
                otm_put_ivs = [p.implied_volatility for p in otm_puts if p.implied_volatility]
                if otm_put_ivs:
                    put_skew = np.mean(otm_put_ivs) - nearest_chain.atm_put_iv

        # Calculate IV vs HV ratio
        iv_vs_hv_ratio = current_iv / historical_volatility if historical_volatility > 0 else None

        return IVAnalysis(
            ticker=ticker,
            current_iv=current_iv,
            historical_volatility=historical_volatility,
            call_skew=call_skew,
            put_skew=put_skew,
            iv_term_structure=iv_term_structure,
            iv_vs_hv_ratio=iv_vs_hv_ratio,
        )

    # ========================================================================
    # STRATEGY P&L CALCULATION
    # ========================================================================

    def calculate_strategy_pnl(
        self,
        strategy: OptionsStrategy,
        price_range: Optional[np.ndarray] = None,
        spot_price: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> List[PnLScenario]:
        """
        Calculate P&L scenarios for a strategy across price range.

        Args:
            strategy: OptionsStrategy to analyze
            price_range: Array of prices to evaluate (auto-generated if None)
            spot_price: Current spot price (required if price_range is None)
            volatility: Historical volatility for range calculation

        Returns:
            List of PnLScenario objects
        """
        # Generate price range if not provided
        if price_range is None:
            if spot_price is None:
                raise ValueError("spot_price required when price_range is None")

            if volatility is None:
                volatility = 0.20  # Default 20% volatility

            # Create price range ±3 standard deviations
            std_dev = spot_price * volatility * np.sqrt(
                strategy.legs[0].contract.days_to_expiration / 365.0
            )
            lower = max(
                OptionsAnalysisParameters.MIN_BREAKEVEN_PRICE,
                spot_price - OptionsAnalysisParameters.PNL_PRICE_RANGE_STD_DEVS * std_dev,
            )
            upper = spot_price + OptionsAnalysisParameters.PNL_PRICE_RANGE_STD_DEVS * std_dev

            price_range = np.linspace(
                lower, upper, OptionsAnalysisParameters.PNL_SIMULATION_POINTS
            )

        scenarios = []

        for price in price_range:
            # Calculate P&L at expiration for this price
            total_pnl = strategy.net_premium  # Start with premium received/paid

            for leg in strategy.legs:
                contract = leg.contract
                quantity = leg.quantity

                # Calculate value at expiration
                if contract.option_type == OptionType.CALL:
                    value_at_exp = max(0.0, price - contract.strike)
                else:  # PUT
                    value_at_exp = max(0.0, contract.strike - price)

                # Add to P&L (multiply by 100 for contract size)
                total_pnl += value_at_exp * quantity * 100

            # Calculate P&L percentage
            capital = abs(strategy.capital_required) if strategy.capital_required != 0 else 1.0
            pnl_percent = (total_pnl / capital) * 100

            scenarios.append(
                PnLScenario(
                    underlying_price=float(price),
                    total_pnl=float(total_pnl),
                    pnl_percent=float(pnl_percent),
                )
            )

        return scenarios

    # ========================================================================
    # STRATEGY DETECTION
    # ========================================================================

    def detect_simple_strategies(
        self, chain: OptionsChain, existing_shares: int = 0
    ) -> List[OptionsStrategy]:
        """
        Detect simple single-leg and two-leg strategies.

        Args:
            chain: OptionsChain to analyze
            existing_shares: Number of shares owned (for covered call detection)

        Returns:
            List of identified OptionsStrategy objects
        """
        strategies = []
        spot = chain.underlying_price

        # Find ATM strike
        all_strikes = list(set([c.strike for c in chain.calls + chain.puts]))
        atm_strike = find_atm_strike(spot, all_strikes)

        if not atm_strike:
            return strategies

        # Get ATM and nearby contracts
        atm_calls = [c for c in chain.calls if c.strike == atm_strike]
        atm_puts = [p for p in chain.puts if p.strike == atm_strike]

        # 1. Long Call (ATM)
        if atm_calls:
            call = atm_calls[0]
            if call.mid_price:
                leg = StrategyLeg(contract=call, quantity=1, action="BUY")
                net_premium = -call.mid_price * 100  # Debit
                capital = abs(net_premium)

                greeks = call.greeks
                strategies.append(
                    OptionsStrategy(
                        strategy_type=StrategyType.LONG_CALL,
                        legs=[leg],
                        net_premium=net_premium,
                        capital_required=capital,
                        max_profit=None,  # Unlimited
                        max_loss=capital,
                        total_delta=greeks.delta if greeks else None,
                        total_gamma=greeks.gamma if greeks else None,
                        total_theta=greeks.theta if greeks else None,
                        total_vega=greeks.vega if greeks else None,
                        description=f"Long {call.strike} Call",
                    )
                )

        # 2. Long Put (ATM)
        if atm_puts:
            put = atm_puts[0]
            if put.mid_price:
                leg = StrategyLeg(contract=put, quantity=1, action="BUY")
                net_premium = -put.mid_price * 100  # Debit
                capital = abs(net_premium)

                greeks = put.greeks
                strategies.append(
                    OptionsStrategy(
                        strategy_type=StrategyType.LONG_PUT,
                        legs=[leg],
                        net_premium=net_premium,
                        capital_required=capital,
                        max_profit=atm_strike * 100,  # Max if stock goes to 0
                        max_loss=capital,
                        total_delta=greeks.delta if greeks else None,
                        total_gamma=greeks.gamma if greeks else None,
                        total_theta=greeks.theta if greeks else None,
                        total_vega=greeks.vega if greeks else None,
                        description=f"Long {put.strike} Put",
                    )
                )

        # 3. Covered Call (if shares owned)
        if existing_shares >= 100 and atm_calls:
            # Find OTM call (higher strike)
            otm_calls = [c for c in chain.calls if c.strike > spot * 1.02]
            if otm_calls:
                call = sorted(otm_calls, key=lambda x: x.strike)[0]  # Nearest OTM
                if call.mid_price:
                    leg = StrategyLeg(contract=call, quantity=-1, action="SELL")
                    net_premium = call.mid_price * 100  # Credit
                    capital = existing_shares * spot  # Stock value

                    greeks = call.greeks
                    delta = -greeks.delta if greeks else -0.5  # Short call

                    strategies.append(
                        OptionsStrategy(
                            strategy_type=StrategyType.COVERED_CALL,
                            legs=[leg],
                            net_premium=net_premium,
                            capital_required=capital,
                            max_profit=(call.strike - spot) * 100 + net_premium,
                            max_loss=capital - net_premium,  # If stock goes to 0
                            total_delta=delta,
                            total_gamma=-greeks.gamma if greeks else None,
                            total_theta=-greeks.theta if greeks else None,
                            total_vega=-greeks.vega if greeks else None,
                            description=f"Covered Call @ {call.strike}",
                        )
                    )

        # 4. Long Straddle (ATM)
        if atm_calls and atm_puts:
            call = atm_calls[0]
            put = atm_puts[0]
            if call.mid_price and put.mid_price:
                leg1 = StrategyLeg(contract=call, quantity=1, action="BUY")
                leg2 = StrategyLeg(contract=put, quantity=1, action="BUY")
                net_premium = -(call.mid_price + put.mid_price) * 100  # Debit
                capital = abs(net_premium)

                call_greeks = call.greeks
                put_greeks = put.greeks

                total_delta = 0.0  # Should be delta-neutral
                if call_greeks and put_greeks:
                    total_delta = call_greeks.delta + put_greeks.delta

                strategies.append(
                    OptionsStrategy(
                        strategy_type=StrategyType.LONG_STRADDLE,
                        legs=[leg1, leg2],
                        net_premium=net_premium,
                        capital_required=capital,
                        max_profit=None,  # Unlimited
                        max_loss=capital,
                        total_delta=total_delta if call_greeks and put_greeks else None,
                        total_gamma=(
                            call_greeks.gamma + put_greeks.gamma
                            if call_greeks and put_greeks
                            else None
                        ),
                        total_theta=(
                            call_greeks.theta + put_greeks.theta
                            if call_greeks and put_greeks
                            else None
                        ),
                        total_vega=(
                            call_greeks.vega + put_greeks.vega
                            if call_greeks and put_greeks
                            else None
                        ),
                        description=f"Long Straddle @ {atm_strike}",
                    )
                )

        return strategies

    def detect_vertical_spreads(self, chain: OptionsChain) -> List[OptionsStrategy]:
        """
        Detect vertical spread strategies (bull/bear call/put spreads).

        Args:
            chain: OptionsChain to analyze

        Returns:
            List of vertical spread strategies
        """
        strategies = []
        spot = chain.underlying_price

        # Bull Call Spread (Buy ATM call, Sell OTM call)
        atm_calls = [c for c in chain.calls if abs(c.strike - spot) / spot < 0.05]
        otm_calls = [c for c in chain.calls if c.strike > spot * 1.05]

        if atm_calls and otm_calls:
            long_call = atm_calls[0]
            short_call = sorted(otm_calls, key=lambda x: x.strike)[0]

            if long_call.mid_price and short_call.mid_price:
                leg1 = StrategyLeg(contract=long_call, quantity=1, action="BUY")
                leg2 = StrategyLeg(contract=short_call, quantity=-1, action="SELL")

                net_premium = -(long_call.mid_price - short_call.mid_price) * 100
                capital = abs(net_premium)
                max_profit = (short_call.strike - long_call.strike) * 100 - capital
                max_loss = capital

                strategies.append(
                    OptionsStrategy(
                        strategy_type=StrategyType.BULL_CALL_SPREAD,
                        legs=[leg1, leg2],
                        net_premium=net_premium,
                        capital_required=capital,
                        max_profit=max_profit,
                        max_loss=max_loss,
                        description=f"Bull Call Spread {long_call.strike}/{short_call.strike}",
                    )
                )

        # Bear Put Spread (Buy ATM put, Sell OTM put)
        atm_puts = [p for p in chain.puts if abs(p.strike - spot) / spot < 0.05]
        otm_puts = [p for p in chain.puts if p.strike < spot * 0.95]

        if atm_puts and otm_puts:
            long_put = atm_puts[0]
            short_put = sorted(otm_puts, key=lambda x: x.strike, reverse=True)[0]

            if long_put.mid_price and short_put.mid_price:
                leg1 = StrategyLeg(contract=long_put, quantity=1, action="BUY")
                leg2 = StrategyLeg(contract=short_put, quantity=-1, action="SELL")

                net_premium = -(long_put.mid_price - short_put.mid_price) * 100
                capital = abs(net_premium)
                max_profit = (long_put.strike - short_put.strike) * 100 - capital
                max_loss = capital

                strategies.append(
                    OptionsStrategy(
                        strategy_type=StrategyType.BEAR_PUT_SPREAD,
                        legs=[leg1, leg2],
                        net_premium=net_premium,
                        capital_required=capital,
                        max_profit=max_profit,
                        max_loss=max_loss,
                        description=f"Bear Put Spread {long_put.strike}/{short_put.strike}",
                    )
                )

        return strategies

    def detect_all_strategies(
        self, chains: List[OptionsChain], existing_shares: int = 0
    ) -> List[OptionsStrategy]:
        """
        Detect all strategies across all chains.

        Args:
            chains: List of OptionsChain objects
            existing_shares: Number of shares owned

        Returns:
            List of all identified strategies
        """
        all_strategies = []

        for chain in chains:
            # Detect simple strategies
            simple = self.detect_simple_strategies(chain, existing_shares)
            all_strategies.extend(simple)

            # Detect vertical spreads
            spreads = self.detect_vertical_spreads(chain)
            all_strategies.extend(spreads)

        # Calculate P&L for each strategy
        for strategy in all_strategies:
            if strategy.legs:
                spot = strategy.legs[0].contract.strike  # Approximation
                vol = strategy.legs[0].contract.implied_volatility or 0.30

                try:
                    pnl_scenarios = self.calculate_strategy_pnl(
                        strategy, spot_price=spot, volatility=vol
                    )
                    strategy.pnl_scenarios = pnl_scenarios

                    # Calculate breakeven points
                    strategy.breakeven_points = self._find_breakevens(pnl_scenarios)

                    # Calculate probability of profit (simple approximation)
                    profitable_scenarios = [s for s in pnl_scenarios if s.total_pnl > 0]
                    if pnl_scenarios:
                        strategy.probability_of_profit = (
                            len(profitable_scenarios) / len(pnl_scenarios)
                        )

                except Exception as e:
                    logger.debug(f"Failed to calculate P&L for strategy: {e}")

        return all_strategies

    def _find_breakevens(self, scenarios: List[PnLScenario]) -> List[float]:
        """
        Find breakeven prices from P&L scenarios.

        Args:
            scenarios: List of PnLScenario objects

        Returns:
            List of breakeven prices where P&L crosses zero
        """
        breakevens = []

        for i in range(len(scenarios) - 1):
            current = scenarios[i]
            next_scenario = scenarios[i + 1]

            # Check if P&L crosses zero
            if current.total_pnl * next_scenario.total_pnl < 0:
                # Linear interpolation to find exact crossing point
                x1, y1 = current.underlying_price, current.total_pnl
                x2, y2 = next_scenario.underlying_price, next_scenario.total_pnl

                # y = mx + b, solve for x when y = 0
                if y2 != y1:
                    breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
                    breakevens.append(breakeven)

        return breakevens
