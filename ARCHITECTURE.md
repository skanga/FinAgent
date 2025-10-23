# Financial Reporting Agent - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                             │
│                                                                         │
│  CLI (main.py)                    Natural Language Input                │
│     │                                      │                            │
│     └──────────────────┬───────────────────┘                            │
└────────────────────────┼────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER                                 │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  FinancialReportOrchestrator (orchestrator.py)                  │    │
│  │  • Coordinates entire pipeline                                  │    │
│  │  • Manages concurrent analysis                                  │    │
│  │  • Error handling with match/case                               │    │
│  │  • Performance tracking                                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                         │                                               │
└─────────────────────────┼───────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┬────────────────┐
          ▼               ▼               ▼                ▼
┌──────────────┐ ┌─────────────┐ ┌──────────────┐ ┌────────────────┐
│  DATA LAYER  │ │ ANALYSIS    │ │ LLM          │ │ VISUALIZATION  │
│              │ │ LAYER       │ │ LAYER        │ │ LAYER          │
└──────────────┘ └─────────────┘ └──────────────┘ └────────────────┘
```

## Detailed Architecture

### 1. Data Layer

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  CachedDataFetcher (fetcher.py)                              │
│  ┌────────────────────────────────────────────────────┐      │
│  │  • fetch_price_history(ticker, period)             │      │
│  │  • Uses @retry decorator for network resilience    │      │
│  │  • Wraps yfinance API                              │      │
│  │  • Validates and enriches data                     │      │
│  └────────────┬───────────────────────────────────────┘      │
│               │                                              │
│               ▼                                              │
│  CacheManager (cache.py)                                     │
│  ┌────────────────────────────────────────────────────┐      │
│  │  • TTL-based caching (default 24h)                 │      │
│  │  • Parquet serialization                           │      │
│  │  • MD5 key generation                              │      │
│  │  • Automatic expiration cleanup                    │      │
│  └────────────────────────────────────────────────────┘      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. Request comes in for ticker data
2. Check cache (key = md5(ticker_period_datatype))
3. If cache hit and not expired → return cached data
4. If cache miss → fetch from yfinance (with retry)
5. Cache result and return

### 2. Analysis Layer

```
┌─────────────────────────────────────────────────────────────┐
│                 FINANCIAL ANALYSIS ENGINE                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  AdvancedFinancialAnalyzer (analyzers.py)                   │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Technical Indicators:                             │     │
│  │  • RSI (14-day)                                    │     │
│  │  • MACD (12/26/9)                                  │     │
│  │  • Bollinger Bands (20-day, 2σ)                    │     │
│  │  • Moving Averages (30/50-day)                     │     │
│  │                                                    │     │
│  │  Risk Metrics:                                     │     │
│  │  • Sharpe Ratio                                    │     │
│  │  • Sortino Ratio                                   │     │
│  │  • Calmar Ratio                                    │     │
│  │  • Treynor Ratio                                   │     │
│  │  • Max Drawdown                                    │     │
│  │  • Value at Risk (95%)                             │     │
│  │  • Beta, Alpha, R²                                 │     │
│  │                                                    │     │
│  │  Fundamentals Parsing:                             │     │
│  │  • Income Statement                                │     │
│  │  • Balance Sheet                                   │     │
│  │  • Cash Flow                                       │     │
│  │  • YoY Growth (4 quarters)                         │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  PortfolioAnalyzer (analyzers.py)                           │
│  ┌────────────────────────────────────────────────────┐     │
│  │  • Correlation matrix                              │     │
│  │  • Diversification ratio                           │     │
│  │  • Concentration risk (Herfindahl index)           │     │
│  │  • Return attribution                              │     │
│  │  • Portfolio optimization                          │     │
│  │  • DataFrame caching (99.5% faster on reuse)       │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. LLM Integration Layer

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM ORCHESTRATION                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  IntegratedLLMInterface (llm_interface.py)                  │
│  ┌────────────────────────────────────────────────────┐     │
│  │                                                    │     │
│  │  Parser Chain:                                     │     │
│  │  Natural Language → ParsedRequest                  │     │
│  │  "Compare AAPL and MSFT" → {tickers, period}       │     │
│  │                                                    │     │
│  │  Narrative Chain:                                  │     │
│  │  Analysis Data → Executive Summary                 │     │
│  │  Generates insights and recommendations            │     │
│  │                                                    │     │
│  │  Review Chain:                                     │     │
│  │  Report → Quality Score (0-10)                     │     │
│  │  Validates accuracy and completeness               │     │
│  │                                                    │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  Provider Support:                                          │
│  • OpenAI (gpt-4o, gpt-4-turbo)                             │
│  • Anthropic (via OpenAI-compatible API)                    │
│  • Google (Gemini)                                          │
│  • Ollama (local models)                                    │
│                                                             │
│  Connection Pooling:                                        │
│  • httpx.Client with max_connections=20                     │
│  • Keep-alive connections                                   │
│  • Timeout handling (30s default)                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4. Visualization Layer

```
┌─────────────────────────────────────────────────────────────┐
│                   CHART GENERATION                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ChartGenerator (charts.py)                                 │
│  ┌────────────────────────────────────────────────────┐     │
│  │  • Price Chart with Technical Overlays             │     │
│  │    - Candlestick/line prices                       │     │
│  │    - RSI subplot                                   │     │
│  │    - Bollinger Bands                               │     │
│  │    - Moving averages                               │     │
│  │                                                    │     │
│  │  • Comparison Chart                                │     │
│  │    - Normalized returns                            │     │
│  │    - Multi-ticker overlay                          │     │
│  │                                                    │     │
│  │  • Risk-Reward Scatter                             │     │
│  │    - Return vs Volatility                          │     │
│  │    - Sharpe ratio sizing                           │     │
│  │                                                    │     │
│  │  Thread-Safe:                                      │     │
│  │  • matplotlib Agg backend                          │     │
│  │  • Explicit figure cleanup                         │     │
│  │  • No gc.collect() (23% faster)                    │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Execution Flow

### Single Ticker Analysis

```
1. User Input
   └─> Validate ticker symbol (utils.validate_ticker_symbol)

2. Data Fetching (with retry)
   └─> Check cache
   └─> If miss: yfinance API call
   └─> Validate DataFrame structure
   └─> Cache result

3. Technical Analysis
   └─> compute_metrics() - adds indicators (RSI, MACD, etc.)
   └─> Uses pandas.assign() for efficiency

4. Risk Analysis
   └─> calculate_advanced_metrics()
   └─> Benchmark comparison if available

5. Fundamentals
   └─> Parse yfinance quarterly data
   └─> Calculate YoY growth

6. Chart Generation (parallel)
   └─> Create technical chart
   └─> Save to output directory

7. Alert System
   └─> Check thresholds (volatility, RSI, drawdown, etc.)

8. Return TickerAnalysis object
```

### Portfolio Analysis

```
1. User Input
   └─> Validate tickers (Pydantic)
   └─> Validate weights (sum = 1.0 ± 0.01)

2. Concurrent Ticker Analysis
   └─> ThreadPoolExecutor (max_workers=3)
   └─> analyze_ticker() for each
   └─> Collect results

3. Portfolio Metrics Calculation
   └─> Load returns data (with caching)
   └─> Calculate correlation matrix
   └─> Compute diversification metrics
   └─> Return attribution analysis

4. Chart Generation
   └─> Comparison chart (normalized returns)
   └─> Risk-reward scatter plot

5. LLM Narrative Generation
   └─> Generate executive summary
   └─> Create detailed insights
   └─> Risk analysis
   └─> Recommendations

6. Quality Review
   └─> LLM reviews report
   └─> Assigns quality score (0-10)
   └─> Identifies issues

7. Save Report
   └─> Markdown file with charts
   └─> JSON quality review
   └─> Performance metrics
```

## Error Handling Architecture

Using Python 3.10+ match/case for clean error categorization:

```python
try:
    # Analysis operations
    ...
except Exception as e:
    match e:
        case ValueError():
            # Validation errors - permanent failure
            # Log as ERROR, don't retry

        case OSError() | ConnectionError() | TimeoutError():
            # Network errors - temporary failure
            # Log as WARNING, could retry

        case KeyError() | pd.errors.ParserError():
            # Data parsing errors - permanent
            # Log as ERROR

        case TypeError():
            # Code/logic errors - permanent
            # Log as ERROR

        case _:
            # Unexpected errors
            # Log with exception() for stack trace
```

**Benefits:**
- Clear error categorization
- Centralized error handling
- Consistent logging levels
- Proper memory cleanup (gc.collect())

## Concurrency Model

```
Main Thread
    │
    ├─> Parse request
    │
    ├─> Fetch benchmark (blocking)
    │
    └─> _analyze_all_tickers()
            │
            ├─> ThreadPoolExecutor (max_workers=3)
            │       │
            │       ├─> Worker 1: analyze_ticker(AAPL)
            │       │       ├─> fetch_price_history (cached)
            │       │       ├─> compute_metrics
            │       │       ├─> calculate_advanced_metrics
            │       │       └─> create_chart
            │       │
            │       ├─> Worker 2: analyze_ticker(MSFT)
            │       │       └─> [same steps]
            │       │
            │       └─> Worker 3: analyze_ticker(GOOGL)
            │               └─> [same steps]
            │
            └─> Collect results (Dict[str, TickerAnalysis])
```

**Thread Safety:**
- matplotlib uses Agg backend (non-interactive, thread-safe)
- Each worker has isolated error handling
- Memory cleanup per worker (gc.collect())
- No shared mutable state
- ProgressTracker uses threading.Lock for thread-safe updates

**Performance Impact:**
- **Sequential:** `time = n_tickers × time_per_ticker`
- **Concurrent:** `time = ceil(n_tickers / max_workers) × time_per_ticker`
- **Speedup:** 2.5x-4.0x depending on worker count and I/O vs CPU ratio

| Tickers | Workers | Sequential | Concurrent | Speedup |
|---------|---------|------------|------------|---------|
| 5       | 3       | 10s        | ~4s        | 2.5x    |
| 10      | 3       | 20s        | ~8s        | 2.5x    |
| 10      | 5       | 20s        | ~5s        | 4.0x    |

**Worker Configuration:**
- Default: `MAX_WORKERS=3` (good balance for most systems)
- Low resource: `MAX_WORKERS=1` (sequential)
- High performance: `MAX_WORKERS=5` (requires good CPU/network)

## Performance Optimizations

### 1. Caching Strategy
- **Cache Key:** `md5(ticker_period_datatype)`
- **TTL:** 24 hours (configurable)
- **Format:** Parquet (tested vs pickle - pickle is faster for our use case)
- **Cleanup:** Automatic on expired entries

### 2. DataFrame Operations
- **pandas.assign()** for chained operations (reduced passes from 10+ to 2-3)
- **Pre-calculation** of reused values
- **Returns caching** in PortfolioAnalyzer (99.5% faster on reuse)

### 3. Chart Generation
- Removed **gc.collect()** after each chart (23% performance improvement)
- Python's automatic GC is sufficient
- Explicit figure cleanup with plt.close()

### 4. Connection Pooling

**LLM Connections (httpx):**
- httpx.Client with persistent HTTP/2 connections
- `max_connections=20` - Total connection limit
- `max_keepalive_connections=10` - Idle connections to maintain
- `keepalive_expiry=30.0` - Keep connections alive for 30 seconds
- **Performance:** 2.25x faster for multiple requests (connection reuse)
- Automatic cleanup via `__del__` method

**yfinance Connections:**
- Uses `curl_cffi` internally for rate limit bypass
- Connection pooling managed automatically by yfinance
- No custom configuration needed
- Caching layer provides primary performance improvement

**Configuration Example:**
```python
llm = IntegratedLLMInterface(
    config,
    max_connections=20,
    max_keepalive_connections=10
)
```

**Benefits:**
- Reduced latency (eliminates TCP/TLS handshake for subsequent requests)
- Better throughput for concurrent requests
- HTTP/2 multiplexing over single connection
- Resource efficiency (limits total connections)

## Configuration Architecture

```python
Config (config.py)
    │
    ├─> Environment Variables (.env)
    │   ├─ OPENAI_API_KEY
    │   ├─ OPENAI_BASE_URL
    │   ├─ OPENAI_MODEL
    │   ├─ MAX_WORKERS
    │   ├─ CACHE_TTL_HOURS
    │   └─ BENCHMARK_TICKER
    │
    ├─> Provider Detection (match/case)
    │   ├─ openai (default)
    │   ├─ anthropic
    │   ├─ google
    │   ├─ ollama (localhost:11434)
    │   └─ local/unknown
    │
    └─> Validation
        ├─ Range checks (max_workers: 1-10, etc.)
        ├─ Type conversion with fallbacks
        └─ Logging of configuration warnings
```

## Data Models

All data structures use **immutable dataclasses** with Pydantic validation:

```
Input Models (Pydantic):
├─ TickerRequest
├─ PortfolioRequest
└─ NaturalLanguageRequest

Analysis Models (Dataclasses):
├─ TickerAnalysis
├─ AdvancedMetrics
├─ TechnicalIndicators
├─ FundamentalData
├─ ComparativeAnalysis
├─ PortfolioMetrics
└─ ReportMetadata
```

**Validation Strategy:**
- Pydantic for external input (CLI, API)
- Centralized ticker validation (utils.validate_ticker_symbol)
- Type hints throughout for static analysis
- Runtime validation with helpful error messages

## Key Design Patterns

1. **Strategy Pattern:** Multiple LLM providers via unified interface
2. **Decorator Pattern:** @retry for network resilience
3. **Factory Pattern:** Error analysis creation
4. **Observer Pattern:** Alert system monitoring metrics
5. **Template Method:** Analysis pipeline with customizable steps
6. **Singleton-like:** Config loaded once from environment

## Security Considerations

- **Path Traversal Prevention:** Output paths validated to be within CWD or home
- **API Key Masking:** First 4 and last 4 chars only in logs
- **Cache Security:** Keys hashed with MD5, no sensitive data in filenames
- **Input Validation:** All ticker symbols sanitized
- **Resource Limits:** Max tickers = 20, max workers = 10

## Monitoring and Observability

**Performance Metrics Tracked:**
- Total execution time
- Cache hit rate
- Success/failure counts per ticker
- Chart generation time
- LLM quality scores
- Error categorization

**Logging Levels:**
- **DEBUG:** Cache operations, provider detection
- **INFO:** Analysis progress, report generation
- **WARNING:** Network errors (retryable)
- **ERROR:** Validation, parsing, type errors
- **EXCEPTION:** Unexpected errors with stack traces

## Extensibility Points

1. **Add New Technical Indicators:**
   - Add method to AdvancedFinancialAnalyzer
   - Update TechnicalIndicators dataclass
   - Modify compute_metrics()

2. **Add New LLM Provider:**
   - Add case to config._get_provider_from_url()
   - Test with OpenAI-compatible API

3. **Add New Chart Types:**
   - Add method to ChartGenerator
   - Use Agg backend for thread safety

4. **Add New Alert Types:**
   - Add threshold to AnalysisThresholds
   - Update AlertSystem.check_alerts()

## File Organization

```
financial_reporting_agent/
├── main.py                    # CLI entry point
├── orchestrator.py            # Main coordinator
├── config.py                  # Configuration management
├── models.py                  # Data models (Pydantic + dataclasses)
├── constants.py               # Configuration constants
├── analyzers.py               # Financial analysis engine
├── fetcher.py                 # Data acquisition with retry
├── cache.py                   # TTL-based caching
├── llm_interface.py           # LLM integration
├── charts.py                  # Visualization
├── alerts.py                  # Alert system
├── utils.py                   # Shared utilities
├── CLAUDE.md                  # Development guide
├── ARCHITECTURE.md            # This file
└── requirements.txt           # Dependencies
```
