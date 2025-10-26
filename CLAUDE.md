# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a financial reporting agent that generates comprehensive financial analysis reports using AI. It combines market data fetching (via yfinance), technical/fundamental analysis, **options analysis (Greeks, IV, strategies)**, portfolio metrics, and LLM-powered narrative generation to produce markdown and HTML reports.

## Architecture

### Core Orchestration Flow

The application follows a pipeline architecture coordinated by `FinancialReportOrchestrator` (orchestrator.py:29):

1. **Input Processing**: Accepts either natural language requests (parsed by LLM) or direct ticker/period parameters
2. **Data Fetching**: CachedDataFetcher retrieves price history with TTL-based caching
3. **Analysis Pipeline**: Each ticker goes through comprehensive analysis
4. **Portfolio Analysis**: Multi-ticker portfolios get correlation, diversification, and risk metrics
5. **Report Generation**: LLM generates narrative summaries and structured markdown
6. **Quality Review**: LLM reviews report for accuracy and assigns quality score

### Key Components

**Orchestrator** (orchestrator.py): Main coordinator that runs the full analysis pipeline. The `run()` method (orchestrator.py:128) is the primary entry point.

**Data Layer**:
- `CachedDataFetcher` (fetcher.py): Wraps yfinance API calls for stock data with intelligent caching
- `OptionsDataFetcher` (options_fetcher.py): Wraps yfinance options API with caching (TTL: 1h)
- `CacheManager` (cache.py): **Dual-format cache** - Parquet for DataFrames, Pickle for Python objects (lists, dataclasses). Stock data TTL: 24h, Options TTL: 1h

**Analysis Layer**:
- `AdvancedFinancialAnalyzer` (analyzers.py:15): Computes technical indicators (RSI, MACD, Bollinger Bands, Stochastic), risk metrics (Sharpe, Sortino, Calmar, Treynor, VaR, max drawdown), and parses fundamentals from yfinance quarterly statements
- `PortfolioAnalyzer` (analyzers.py:251): Calculates portfolio-level metrics including diversification ratio, correlation matrix, concentration risk (Herfindahl index), and return attribution
- `OptionsAnalyzer` (options_analyzer.py): Calculates Greeks (Black-Scholes-Merton), solves for IV (Newton-Raphson + Brent's), detects 18+ strategies, generates P&L scenarios
- `PortfolioOptionsAnalyzer` (analyzers.py): Aggregates portfolio-level Greeks, generates hedging recommendations, analyzes concentration risk

**LLM Integration** (llm_interface.py):
- Uses LangChain with ChatOpenAI
- Three main chains: parser (natural language → structured params), narrative (data → executive summary), review (report quality check)
- All prompts are in `_setup_prompts()` (llm_interface.py:31)

**Visualization** (charts.py):
- Thread-safe matplotlib chart generation with Agg backend
- Creates price charts with technical overlays, comparison charts (normalized returns), and risk-reward scatter plots

**Alert System** (alerts.py): Rule-based monitoring for high volatility, large drawdowns, extreme RSI, and benchmark underperformance

### Data Models

All data structures are immutable dataclasses in models.py:
- `TickerAnalysis`: Complete analysis results for a single ticker
- `PortfolioMetrics`: Portfolio-level aggregated metrics
- `AdvancedMetrics`: Risk-adjusted performance metrics
- `FundamentalData`: Parsed financial statements (income, balance, cash flow)
- `ParsedRequest`: Validated natural language request
- `ReportMetadata`: Final report with all artifacts and performance metrics

### Configuration

Configuration is loaded from environment variables via `Config.from_env()` (config.py:43):
- `OPENAI_API_KEY` (required)
- `OPENAI_BASE_URL` (default: https://api.openai.com/v1)
- `OPENAI_MODEL` (default: gpt-4o)
- `MAX_WORKERS` (default: 3)
- `CACHE_TTL_HOURS` (default: 24)
- `BENCHMARK_TICKER` (default: SPY)
- `GENERATE_HTML` (default: true) - Enable/disable HTML report generation
- `EMBED_IMAGES_IN_HTML` (default: false) - Embed images as base64 in HTML
- `OPEN_IN_BROWSER` (default: true) - Prompt to open HTML report in browser (default: yes)

All values have validation in `__post_init__()`.

## Running the Application

**Basic usage:**
```bash
# Set required environment variable first
export OPENAI_API_KEY="your-key-here"

# Natural language request
python main.py --request "Compare AAPL and MSFT over the past year"

# Direct parameters
python main.py --tickers AAPL,MSFT,GOOGL --period 1y

# With portfolio weights
python main.py --tickers AAPL,MSFT,GOOGL --weights 0.5,0.3,0.2

# Custom output directory
python main.py --tickers TSLA,NVDA --period 6mo --output ./my_reports

# Default run (AAPL, MSFT, GOOGL for 1y)
python main.py
```

**HTML Report Options:**
```bash
# Disable HTML generation (markdown only)
python main.py --tickers AAPL --no-html

# Don't open in browser automatically
python main.py --tickers AAPL --no-browser

# Embed images as base64 (single-file HTML)
python main.py --tickers AAPL --embed-images
```

**Options Analysis:**
```bash
# Enable options analysis
python main.py --tickers AAPL --options

# Analyze more expirations (default: 3, max: 10)
python main.py --tickers SPY --options --options-expirations 5

# Portfolio with options
python main.py --tickers AAPL,MSFT --period 1y --options
```

**Utility commands:**
```bash
# Clear cache
python main.py --clear-cache

# Verbose logging
python main.py --verbose -t AAPL -p 1y
```

**Valid periods:** 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

## Testing

The project includes 600+ comprehensive tests organized in the `tests/` directory:

**Running Tests:**
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_analyzers_unit.py -v

# Run by category
pytest tests/ -m unit              # Fast unit tests
pytest tests/ -m integration       # Integration tests
pytest tests/ -m "not slow"        # Skip slow tests

# With coverage
pytest tests/ --cov=. --cov-report=html
```

**Test Organization:**
- `tests/conftest.py` - Shared fixtures (configs, mock data, temp directories)
- `tests/test_*.py` - Test modules (26 files, 600+ tests total)
- Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`, `@pytest.mark.network`, `@pytest.mark.llm`

**Key Test Files:**
- `test_analyzers_unit.py` - Financial metrics calculations
- `test_llm_interface.py` - LLM integration and prompts
- `test_orchestrator.py` - End-to-end workflow tests
- `test_options_fetcher.py` - Options data fetching
- `test_html_generation.py` - HTML report generation
- `test_memory_leaks.py` - Memory profiling
- `test_performance.py` - Performance benchmarks

See `docs/TESTING_QUICK_START.md` for detailed testing guide and `pytest.ini` for configuration.

## Development Notes

**Adding new technical indicators:** Add computation method to `AdvancedFinancialAnalyzer`, update `compute_metrics()` to calculate it, and add to `TechnicalIndicators` dataclass.

**Adding new fundamental metrics:** Extend `parse_fundamentals()` (analyzers.py:183) to extract from yfinance quarterly statements. Add fields to `FundamentalData` dataclass.

**Modifying LLM prompts:** Update templates in `IntegratedLLMInterface._setup_prompts()` (llm_interface.py:31). Remember to rebuild chains in `_setup_chains()` if you add new prompt templates.

**Caching behavior:** Cache is keyed by `md5(ticker_period_datatype)`. TTL is checked on read. Use `cache.clear_expired()` to manually clean. Cache directory is `./.cache/` by default.

**Error handling:** Individual ticker failures don't abort the full run. Failed analyses are tracked in `ReportMetadata.analyses` with `error` field set. Portfolio analysis requires at least 2 successful ticker analyses.

**Performance tracking:** The orchestrator tracks execution time, success/failure counts, chart generation, and LLM quality scores in `performance_metrics` dict (orchestrator.py:237).

**Report structure:** Generated markdown always includes: Executive Summary (LLM), Portfolio Overview, Key Metrics Table, Fundamental Analysis, Detailed Stock Analysis, Risk Analysis, Recommendations, Active Alerts, and Data Quality Notes.

**HTML report generation:** By default, reports are generated in both markdown and HTML formats. HTML reports use a professional, responsive template with styled charts and tables. After generation, the user is prompted to open the HTML report in their browser (press Enter to accept, or type 'n' to skip). This can be disabled with `--no-browser`. HTML generation uses a custom regex-based markdown parser (html_generator.py) with no additional dependencies. Images can be embedded as base64 or referenced relatively.

## Important Implementation Details

**yfinance data structure:** The fetcher expects yfinance history DataFrames with Date, Open, High, Low, Close, Volume columns. It resets the index to make Date a column and adds a ticker column.

**Benchmark alignment:** When calculating beta/alpha/information ratio, returns are aligned to common dates between ticker and benchmark using pandas index intersection (analyzers.py:114).

**Portfolio weights:** Must sum to 1.0 (±0.01 tolerance). If not provided, defaults to equal weights. Validated in `PortfolioAnalyzer.calculate_portfolio_metrics()` (analyzers.py:271).

**Thread safety:** matplotlib is set to 'Agg' backend (charts.py:6) for thread-safe chart generation. Each chart closes all figures after saving.

**LLM retry logic:** Natural language parsing retries up to `max_retries` times (default 3). Handles both JSON parse errors and validation errors. See llm_interface.py:101.

**Progress tracking:** Uses `ProgressTracker` (utils.py) for long-running operations, displaying elapsed time, current item, and ETA.

## Options Analysis Feature

**Architecture:** Options analysis is **optional** and disabled by default (no performance impact when not used).

**Data Models** (models_options.py):
- `OptionsContract`: Single options contract with strike, expiration, prices, Greeks
- `OptionsChain`: Complete chain for one expiration (calls + puts)
- `OptionsStrategy`: Detected strategy with legs, P&L, risk metrics
- `TickerOptionsAnalysis`: Complete options analysis for a ticker
- `PortfolioOptionsMetrics`: Portfolio-level aggregated Greeks

**Greeks Calculation** (options_analyzer.py):
- Uses Black-Scholes-Merton model for European-style options
- Calculates Delta, Gamma, Theta, Vega, Rho with dollar equivalents
- Fast, accurate for typical equity options analysis

**IV Solver** (options_analyzer.py):
- Newton-Raphson method (fast convergence, typically <10 iterations)
- Falls back to Brent's method if Newton-Raphson fails
- Handles edge cases (deep ITM/OTM, near expiration)

**Strategy Detection** (options_analyzer.py):
- Pattern-based detection (not ML-based)
- Detects 18+ strategies: long/short calls/puts, straddles/strangles, vertical/horizontal/diagonal spreads, iron condors, butterflies, calendar spreads
- Extensible via `detect_all_strategies()` method

**Caching Behavior:**
- Options chains cached for 1 hour (configurable via `OPTIONS_CACHE_TTL_HOURS`)
- Separate from stock data cache (which uses 24h TTL)
- Uses Pickle format (not Parquet) for OptionsChain objects

**Visualizations** (charts.py):
- Options chain heatmaps (volume/OI/IV across strikes and expirations)
- Greeks visualization (multi-panel Delta/Gamma/Theta/Vega charts)
- P&L diagrams (payoff curves with breakevens and profit zones)
- IV surface/skew plots (3D surface or 2D skew)

**LLM Integration:**
- Three new prompts: options narrative, strategy recommendations, portfolio hedging
- Temperature: 0.3 for consistency
- Includes specific numbers (strikes, premiums, Greeks) in narratives

**Adding Options Support:**
1. Enable via `--options` flag or `INCLUDE_OPTIONS=true` env var
2. Orchestrator checks `portfolio_request.include_options` flag
3. If enabled, runs `_analyze_options_for_ticker()` for each ticker
4. Results stored in `ticker_analysis.options_analysis`
5. Portfolio-level aggregation in `PortfolioOptionsAnalyzer`

**Performance:**
- Typical analysis time: 10-15 seconds per ticker (first run)
- With caching: 5-7 seconds per ticker (subsequent runs)
- Concurrent analysis with ThreadPoolExecutor (default: 3 workers)
- Graceful degradation: Individual ticker options failures don't abort run
