# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a financial reporting agent that generates comprehensive financial analysis reports using AI. It combines market data fetching (via yfinance), technical/fundamental analysis, portfolio metrics, and LLM-powered narrative generation to produce markdown reports.

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
- `CachedDataFetcher` (fetcher.py): Wraps yfinance API calls with intelligent caching
- `CacheManager` (cache.py): Pickle-based cache with TTL (default 24h), keyed by ticker+period+data_type

**Analysis Layer**:
- `AdvancedFinancialAnalyzer` (analyzers.py:15): Computes technical indicators (RSI, MACD, Bollinger Bands), risk metrics (Sharpe, Sortino, Calmar, Treynor, VaR, max drawdown), and parses fundamentals from yfinance quarterly statements
- `PortfolioAnalyzer` (analyzers.py:251): Calculates portfolio-level metrics including diversification ratio, correlation matrix, concentration risk (Herfindahl index), and return attribution

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

**Utility commands:**
```bash
# Clear cache
python main.py --clear-cache

# Verbose logging
python main.py --verbose -t AAPL -p 1y
```

**Valid periods:** 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

## Development Notes

**Adding new technical indicators:** Add computation method to `AdvancedFinancialAnalyzer`, update `compute_metrics()` to calculate it, and add to `TechnicalIndicators` dataclass.

**Adding new fundamental metrics:** Extend `parse_fundamentals()` (analyzers.py:183) to extract from yfinance quarterly statements. Add fields to `FundamentalData` dataclass.

**Modifying LLM prompts:** Update templates in `IntegratedLLMInterface._setup_prompts()` (llm_interface.py:31). Remember to rebuild chains in `_setup_chains()` if you add new prompt templates.

**Caching behavior:** Cache is keyed by `md5(ticker_period_datatype)`. TTL is checked on read. Use `cache.clear_expired()` to manually clean. Cache directory is `./.cache/` by default.

**Error handling:** Individual ticker failures don't abort the full run. Failed analyses are tracked in `ReportMetadata.analyses` with `error` field set. Portfolio analysis requires at least 2 successful ticker analyses.

**Performance tracking:** The orchestrator tracks execution time, success/failure counts, chart generation, and LLM quality scores in `performance_metrics` dict (orchestrator.py:237).

**Report structure:** Generated markdown always includes: Executive Summary (LLM), Portfolio Overview, Key Metrics Table, Fundamental Analysis, Detailed Stock Analysis, Risk Analysis, Recommendations, Active Alerts, and Data Quality Notes.

## Important Implementation Details

**yfinance data structure:** The fetcher expects yfinance history DataFrames with Date, Open, High, Low, Close, Volume columns. It resets the index to make Date a column and adds a ticker column.

**Benchmark alignment:** When calculating beta/alpha/information ratio, returns are aligned to common dates between ticker and benchmark using pandas index intersection (analyzers.py:114).

**Portfolio weights:** Must sum to 1.0 (±0.01 tolerance). If not provided, defaults to equal weights. Validated in `PortfolioAnalyzer.calculate_portfolio_metrics()` (analyzers.py:271).

**Thread safety:** matplotlib is set to 'Agg' backend (charts.py:6) for thread-safe chart generation. Each chart closes all figures after saving.

**LLM retry logic:** Natural language parsing retries up to `max_retries` times (default 3). Handles both JSON parse errors and validation errors. See llm_interface.py:101.

**Progress tracking:** Uses `ProgressTracker` (utils.py) for long-running operations, displaying elapsed time, current item, and ETA.
