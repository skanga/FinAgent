"""
CLI entry point for the financial reporting agent.
"""
import sys
import logging
import argparse
from typing import List, Optional

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from config import Config
from orchestrator import FinancialReportOrchestrator
from constants import MAX_TICKERS_ALLOWED, PORTFOLIO_WEIGHT_TOLERANCE, VALID_PERIODS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_cli() -> argparse.ArgumentParser:
    """
    Sets up the command-line interface for the financial reporting agent.

    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(
        description='Advanced Financial Report Generator with AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [options]",
        add_help=True,
        epilog="""
Examples:
  # Natural language request
  %(prog)s --request "Compare AAPL and MSFT over the past year"
  
  # Direct parameters
  %(prog)s --tickers AAPL,MSFT,GOOGL --period 1y
  
  # Custom output directory
  %(prog)s --tickers TSLA,NVDA --period 6mo --output ./my_report
  
  # With custom portfolio weights
  %(prog)s --tickers AAPL,MSFT,GOOGL --weights 0.5,0.3,0.2
        """
    )
    
    parser.add_argument(
        '--request', '-r',
        type=str,
        help='Natural language request (e.g., "Analyze tech stocks over 6 months")'
    )
    
    parser.add_argument(
        '--tickers', '-t',
        type=str,
        help='Comma-separated ticker symbols (e.g., "AAPL,MSFT,GOOGL")'
    )
    
    parser.add_argument(
        '--period', '-p',
        type=str,
        choices=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
        default='1y',
        help='Analysis period (default: 1y)'
    )
    
    parser.add_argument(
        '--weights', '-w',
        type=str,
        help='Comma-separated portfolio weights (e.g., "0.4,0.3,0.3")'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default="./financial_reports",
        help='Output directory (default: ./financial_reports)'
    )
    
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear cache before running'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def validate_tickers(tickers: List[str]) -> None:
    """
    Validates a list of ticker symbols.

    Args:
        tickers (List[str]): The list of ticker symbols to validate.

    Raises:
        ValueError: If the list of tickers is invalid.
    """
    if not tickers:
        raise ValueError("At least one ticker symbol is required")

    if len(tickers) > MAX_TICKERS_ALLOWED:
        raise ValueError(f"Too many tickers ({len(tickers)}). Maximum {MAX_TICKERS_ALLOWED} allowed to avoid API rate limits")

    # Validate ticker format (basic checks)
    invalid_tickers = []
    for ticker in tickers:
        # Check length (most tickers are 1-5 characters)
        if len(ticker) < 1 or len(ticker) > 10:
            invalid_tickers.append(f"{ticker} (invalid length)")
            continue

        # Check for valid characters (letters, numbers, dots, hyphens)
        if not all(c.isalnum() or c in '.-' for c in ticker):
            invalid_tickers.append(f"{ticker} (invalid characters)")
            continue

        # Warn about potentially problematic tickers
        if ticker.lower() in ['test', 'null', 'none', 'undefined']:
            invalid_tickers.append(f"{ticker} (suspicious name)")

    if invalid_tickers:
        raise ValueError(f"Invalid ticker symbols: {', '.join(invalid_tickers)}")


def validate_period(period: str) -> None:
    """
    Validates that the period is a supported yfinance period.

    Args:
        period (str): The period to validate.

    Raises:
        ValueError: If the period is invalid.
    """
    if period not in VALID_PERIODS:
        raise ValueError(
            f"Invalid period '{period}'. Must be one of: {', '.join(VALID_PERIODS)}"
        )


def parse_weights(weights_str: Optional[str], tickers: List[str]) -> Optional[dict]:
    """
    Parses a string of comma-separated portfolio weights.

    Args:
        weights_str (Optional[str]): The string of weights to parse.
        tickers (List[str]): The list of tickers.

    Returns:
        Optional[dict]: A dictionary of tickers and weights, or None if no weights are provided.

    Raises:
        ValueError: If the weights are invalid.
    """
    if not weights_str:
        return None

    try:
        # Parse weights
        weights_list = [float(w.strip()) for w in weights_str.split(',')]

        # Validate count matches tickers
        if len(weights_list) != len(tickers):
            raise ValueError(
                f"Number of weights ({len(weights_list)}) must match number of tickers ({len(tickers)})"
            )

        # Validate individual weights BEFORE checking sum
        # This provides better error messages
        invalid_weights = []
        for ticker, weight in zip(tickers, weights_list):
            if weight < 0:
                invalid_weights.append(f"{ticker}={weight} (negative)")
            elif weight > 1:
                invalid_weights.append(f"{ticker}={weight} (>1.0)")

        if invalid_weights:
            raise ValueError(
                f"All weights must be between 0 and 1. Invalid: {', '.join(invalid_weights)}"
            )

        # Validate sum
        total = sum(weights_list)
        if abs(total - 1.0) > PORTFOLIO_WEIGHT_TOLERANCE:
            raise ValueError(
                f"Weights must sum to 1.0 (±{PORTFOLIO_WEIGHT_TOLERANCE}), got {total:.4f}"
            )

        return {ticker: weight for ticker, weight in zip(tickers, weights_list)}

    except ValueError as e:
        # If already a ValueError with our message, re-raise as-is
        if str(e).startswith(("Number of weights", "All weights", "Weights must sum")):
            raise
        # Otherwise, wrap in more descriptive message
        raise ValueError(f"Invalid weights format: {e}")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = setup_cli()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load and validate config
        config = Config.from_env()
        
        # Clear cache if requested
        if args.clear_cache:
            from cache import CacheManager
            cache = CacheManager(ttl_hours=config.cache_ttl_hours)
            cleared = cache.clear_expired()
            print(f"🗑️  Cleared {cleared} cache entries\n")
        
        # Create orchestrator
        orchestrator = FinancialReportOrchestrator(config)
        
        print("=" * 42)
        print("FINANCIAL REPORT GENERATOR")
        print("=" * 42)
        
        # Determine execution mode
        if args.request:
            # Natural language mode
            print(f"📝 Request: {args.request}")
            print("=" * 42)
            result = orchestrator.run_from_natural_language(args.request, args.output)

        elif args.tickers:
            # Direct mode
            tickers = [t.strip().upper() for t in args.tickers.split(',')]

            # Validate inputs before processing
            validate_tickers(tickers)
            validate_period(args.period)

            # Parse weights if provided
            weights = parse_weights(args.weights, tickers) if args.weights else None

            print(f"📊 Tickers: {', '.join(tickers)}")
            print(f"⏰ Period: {args.period}")
            if weights:
                print(f"⚖️  Weights: {weights}")
            print("=" * 60)

            result = orchestrator.run(tickers, args.period, args.output, weights)

        else:
            # Default mode
            default_tickers = ['AAPL', 'MSFT', 'GOOGL']
            validate_tickers(default_tickers)
            validate_period(args.period)

            print("📊 Using default tickers: AAPL, MSFT, GOOGL")
            print(f"⏰ Period: {args.period}")
            print("=" * 60)
            result = orchestrator.run(default_tickers, args.period, args.output)
        
        # Print summary
        print("\n" + "=" * 60)
        print("REPORT GENERATION COMPLETE")
        print("=" * 60)
        print(f"📄 Report: {result.final_markdown_path}")
        print(f"⏱️ Time: {result.performance_metrics['execution_time_seconds']}s")
        print(f"✅ Successful: {result.performance_metrics['successful']}")
        print(f"❌ Failed: {result.performance_metrics['failed']}")
        print(f"🖼️ Charts: {result.performance_metrics['charts_generated']}")
        print(f"⭐ Quality: {result.performance_metrics['quality_score']}/10")
        
        if result.portfolio_metrics:
            print(f"\n💼 Portfolio Analysis:")
            print(f"   Value: ${result.portfolio_metrics.total_value:,.2f}")
            print(f"   Return: {result.portfolio_metrics.portfolio_return*100:.2f}%")
            print(f"   Volatility: {result.portfolio_metrics.portfolio_volatility*100:.2f}%")
            if result.portfolio_metrics.portfolio_sharpe:
                print(f"   Sharpe: {result.portfolio_metrics.portfolio_sharpe:.2f}")
            if result.portfolio_metrics.diversification_ratio:
                print(f"   Diversification: {result.portfolio_metrics.diversification_ratio:.2f}")
        
        if result.review_issues:
            print(f"\n⚠️  Review Issues ({len(result.review_issues)}):")
            for issue in result.review_issues:
                print(f"  • {issue}")
        
        print("\n" + "=" * 60)
        print("✨ Completed:")
        print("  ✅ Financial ratios computed (P/E, ROE, margins, etc.)")
        print("  ✅ Fundamental data parsed (revenue, earnings, cash flow)")
        print("  ✅ Portfolio-level metrics (diversification, correlation)")
        print("  ✅ LLM-powered narratives and insights")
        print("  ✅ Natural language request parsing")
        print("  ✅ Intelligent caching with TTL")
        print("  ✅ Report quality review")
        print("  ✅ Progress tracking")
        print("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        return 1

    except Exception as e:  # Broad catch is intentional here - top-level error handler
        logger.error(f"Fatal error: {e}", exc_info=args.verbose)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
