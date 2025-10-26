"""
CLI entry point for the financial reporting agent.
"""

import sys
import logging
import argparse
import webbrowser
from typing import List, Optional, Dict

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

from config import Config
from orchestrator import FinancialReportOrchestrator
from models import PortfolioRequest
from pydantic import ValidationError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_cli() -> argparse.ArgumentParser:
    """
    Sets up the command-line interface for the financial reporting agent.

    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Advanced Financial Report Generator with AI",
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
        """,
    )

    parser.add_argument(
        "--request",
        "-r",
        type=str,
        help='Natural language request (e.g., "Analyze tech stocks over 6 months")',
    )

    parser.add_argument(
        "--tickers",
        "-t",
        type=str,
        help='Comma-separated ticker symbols (e.g., "AAPL,MSFT,GOOGL")',
    )

    parser.add_argument(
        "--period",
        "-p",
        type=str,
        choices=[
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
        ],
        default="1y",
        help="Analysis period (default: 1y)",
    )

    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        help='Comma-separated portfolio weights (e.g., "0.4,0.3,0.3")',
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./financial_reports",
        help="Output directory (default: ./financial_reports)",
    )

    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear cache before running"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Disable HTML report generation (only generate markdown)",
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open HTML report in browser automatically",
    )

    parser.add_argument(
        "--embed-images",
        action="store_true",
        help="Embed images as base64 in HTML (creates larger single-file report)",
    )

    # Options analysis arguments
    parser.add_argument(
        "--options",
        "--with-options",
        action="store_true",
        help="Include options analysis in the report (Greeks, strategies, IV analysis)",
    )

    parser.add_argument(
        "--options-expirations",
        type=int,
        default=3,
        metavar="N",
        help="Number of options expirations to analyze (default: 3, max: 10)",
    )

    return parser


def parse_weights(weights_str: Optional[str], tickers: List[str]) -> Optional[Dict[str, float]]:
    """
    Parses a string of comma-separated portfolio weights into a dictionary.

    Args:
        weights_str (Optional[str]): The string of weights to parse (e.g., "0.5,0.3,0.2").
        tickers (List[str]): The list of tickers (must match weight count).

    Returns:
        Optional[Dict[str, float]]: A dictionary mapping tickers to weights, or None if no weights provided.

    Raises:
        ValueError: If weights format is invalid or count doesn't match tickers.
    """
    if not weights_str:
        return None

    try:
        # Parse weights
        weights_list = [float(w.strip()) for w in weights_str.split(",")]

        # Validate count matches tickers
        if len(weights_list) != len(tickers):
            raise ValueError(
                f"Number of weights ({len(weights_list)}) must match number of tickers ({len(tickers)})"
            )

        # Create dictionary (Pydantic will validate sum and individual values)
        return {ticker: weight for ticker, weight in zip(tickers, weights_list)}

    except ValueError as e:
        # If already a ValueError with our message, re-raise as-is
        if "Number of weights" in str(e):
            raise
        # Otherwise, wrap in more descriptive message
        raise ValueError(f"Invalid weights format: {e}") from e


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

        # Override HTML settings from CLI flags
        if args.no_html:
            config.generate_html = False
        if args.no_browser:
            config.open_in_browser = False
        if args.embed_images:
            config.embed_images_in_html = True

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
            # Direct mode - use Pydantic validation (which uses centralized utils.validate_ticker_list)
            tickers = [t.strip() for t in args.tickers.split(",")]

            # Parse weights if provided
            weights = parse_weights(args.weights, tickers) if args.weights else None

            # Create and validate request using Pydantic
            # Note: Pydantic validators will normalize (uppercase) and validate tickers
            try:
                portfolio_request = PortfolioRequest(
                    tickers=tickers,
                    period=args.period,
                    weights=weights,
                    include_options=args.options,
                    options_expirations=min(args.options_expirations, 10)  # Cap at 10
                )
            except ValidationError as e:
                # Extract user-friendly error messages from Pydantic
                error_messages = []
                for error in e.errors():
                    field = " -> ".join(str(loc) for loc in error['loc'])
                    msg = error['msg']
                    error_messages.append(f"{field}: {msg}")
                raise ValueError("\n".join(error_messages)) from e

            print(f"📊 Tickers: {', '.join(portfolio_request.tickers)}")
            print(f"⏰ Period: {portfolio_request.period}")
            if portfolio_request.weights:
                print(f"⚖️  Weights: {portfolio_request.weights}")
            if portfolio_request.include_options:
                print(f"📈 Options: ENABLED ({portfolio_request.options_expirations} expirations)")
            print("=" * 60)

            result = orchestrator.run(portfolio_request, args.output)

        else:
            # Default mode - use Pydantic validation
            try:
                portfolio_request = PortfolioRequest(
                    tickers=["AAPL", "MSFT", "GOOGL"],
                    period=args.period,
                    weights=None,
                    include_options=args.options,
                    options_expirations=min(args.options_expirations, 10)
                )
            except ValidationError as e:
                # Extract user-friendly error messages
                error_messages = []
                for error in e.errors():
                    field = " -> ".join(str(loc) for loc in error['loc'])
                    msg = error['msg']
                    error_messages.append(f"{field}: {msg}")
                raise ValueError("\n".join(error_messages)) from e

            print("📊 Using default tickers: AAPL, MSFT, GOOGL")
            print(f"⏰ Period: {portfolio_request.period}")
            if portfolio_request.include_options:
                print(f"📈 Options: ENABLED ({portfolio_request.options_expirations} expirations)")
            print("=" * 60)
            result = orchestrator.run(portfolio_request, args.output)

        # Open HTML in browser if enabled (with user confirmation for security)
        browser_opened = False
        if result.final_html_path and config.open_in_browser:
            try:
                # Prompt for confirmation before opening browser (security best practice)
                response = input(f"Open report in browser? ({result.final_html_path}) [Y/n]: ").strip()
                # Default to 'yes' if user just presses Enter (empty response)
                if response == '' or response.lower() in ('y', 'yes'):
                    # Convert to file:// URL for proper browser handling
                    file_url = result.final_html_path.as_uri()
                    webbrowser.open(file_url, new=2)  # new=2 opens in new tab if possible
                    browser_opened = True
            except Exception as e:
                logger.warning(f"Failed to open browser: {e}")

        # Print summary
        print("\n" + "=" * 60)
        print("REPORT GENERATION COMPLETE")
        print("=" * 60)
        print(f"📄 Markdown: {result.final_markdown_path}")
        if result.final_html_path:
            print(f"🌐 HTML: {result.final_html_path}")
            if browser_opened:
                print("   ✓ Opened in browser")
        print(f"⏱️  Time: {result.performance_metrics['execution_time_seconds']}s")
        print(f"✅ Successful: {result.performance_metrics['successful']}")
        print(f"❌ Failed: {result.performance_metrics['failed']}")
        print(f"🖼️  Charts: {result.performance_metrics['charts_generated']}")
        print(f"⭐ Quality: {result.performance_metrics['quality_score']}/10")

        if result.portfolio_metrics:
            print("\n💼 Portfolio Analysis:")
            print(f"   Value: ${result.portfolio_metrics.total_value:,.2f}")
            print(f"   Return: {result.portfolio_metrics.portfolio_return*100:.2f}%")
            print(
                f"   Volatility: {result.portfolio_metrics.portfolio_volatility*100:.2f}%"
            )
            if result.portfolio_metrics.portfolio_sharpe:
                print(f"   Sharpe: {result.portfolio_metrics.portfolio_sharpe:.2f}")
            if result.portfolio_metrics.diversification_ratio:
                print(
                    f"   Diversification: {result.portfolio_metrics.diversification_ratio:.2f}"
                )

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
