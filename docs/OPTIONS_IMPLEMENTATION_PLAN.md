# Options Analysis Integration Plan

## Overview
Add **advanced options analysis** with full strategy identification, Greeks calculation, portfolio-level integration, and comprehensive visualizations. Implementation follows the existing architectural patterns with optional CLI flag activation.

---

## Phase 1: Data Models & Core Structure

### 1.1 Create `models_options.py`
New file containing immutable dataclasses for options data:
- **`OptionsContract`**: Individual contract details (strike, expiry, type, bid/ask, volume, OI, IV)
- **`GreeksData`**: Delta, Gamma, Theta, Vega, Rho for a contract
- **`OptionsChain`**: Complete chain for an expiration date
- **`OptionsStrategy`**: Identified strategy with legs, P&L scenarios, risk metrics
- **`TickerOptionsAnalysis`**: Full options analysis for one ticker (chains, strategies, recommendations)
- **`PortfolioOptionsMetrics`**: Aggregated portfolio-level Greeks, hedging suggestions, exposure analysis

### 1.2 Update `models.py`
- Add `options_analysis: Optional[TickerOptionsAnalysis]` field to `TickerAnalysis` dataclass
- Add `options_metrics: Optional[PortfolioOptionsMetrics]` field to `PortfolioMetrics` dataclass

### 1.3 Update `constants.py`
Add new `OptionsAnalysisParameters` class:
- IV calculation methods, Greeks calculation defaults
- Strategy detection thresholds (e.g., max spread width, min contracts)
- P&L simulation parameters (price ranges, time steps)
- Visualization parameters (heatmap resolution, surface smoothing)

---

## Phase 2: Data Fetching Layer

### 2.1 Create `options_fetcher.py`
**`OptionsDataFetcher`** class inheriting from caching patterns:
- `fetch_options_chain(ticker: str, expiration: Optional[date]) -> OptionsChain`
- `fetch_all_expirations(ticker: str) -> List[OptionsChain]`
- Integration with `CachedDataFetcher` and `CacheManager` for TTL-based caching
- Use yfinance's `ticker.options` and `ticker.option_chain()` APIs
- Retry logic using tenacity decorator (matching `fetcher.py` pattern)
- Connection pooling via shared session

---

## Phase 3: Analysis Engine

### 3.1 Create `options_analyzer.py`
**`OptionsAnalyzer`** class with methods:

**Greeks Calculation:**
- `calculate_greeks(contract: OptionsContract, spot: float, rate: float) -> GreeksData`
- Black-Scholes-Merton model implementation
- Support for American options using binomial tree approximation

**Implied Volatility:**
- `calculate_iv(contract: OptionsContract, spot: float) -> float`
- Newton-Raphson iterative solver
- IV surface/skew detection across strikes

**Strategy Detection:**
- `detect_strategies(chains: List[OptionsChain], positions: Optional[Dict]) -> List[OptionsStrategy]`
- Pattern matching for: covered calls, protective puts, straddles, strangles, spreads (vertical, calendar, diagonal), iron condors, butterflies
- Use existing positions if provided (from portfolio)

**Risk Analysis:**
- `calculate_strategy_pnl(strategy: OptionsStrategy, price_range: np.array) -> pd.DataFrame`
- Monte Carlo simulation for expected returns
- Max profit/loss, breakeven points, probability of profit

**Integration with Stock Analysis:**
- Compare IV to historical volatility (from `AdvancedMetrics`)
- Hedging recommendations based on portfolio Greeks exposure
- Earnings/event risk detection (IV spikes)

---

## Phase 4: Visualization Layer

### 4.1 Extend `charts.py`
Add new methods to `ThreadSafeChartGenerator`:

**`create_options_chain_heatmap()`**
- 2D heatmap: strikes (Y) × expirations (X)
- Three variants: volume, open interest, implied volatility
- Color gradients with annotation

**`create_greeks_visualization()`**
- Multi-panel chart showing all Greeks
- Sensitivity curves (Greeks vs. underlying price)
- Portfolio-level Greeks exposure bar chart

**`create_pnl_diagram()`**
- Classic payoff diagram at expiration
- Multiple scenarios (current, ±1σ, ±2σ)
- Breakeven markers and max profit/loss annotations

**`create_iv_surface()`**
- 3D surface plot (strike, time, IV) using matplotlib 3D
- 2D skew plot (strike vs IV) for near-term expiration
- ATM IV term structure

All methods follow existing thread-safe pattern with `Agg` backend.

---

## Phase 5: Portfolio Integration

### 5.1 Extend `analyzers.py`
Add **`PortfolioOptionsAnalyzer`** class:
- Aggregate Greeks across all portfolio positions
- Calculate portfolio-level hedging ratios
- Identify concentration risks in options exposure
- Suggest balancing strategies (e.g., "Portfolio is net short Vega, consider long volatility hedge")

### 5.2 Update `PortfolioAnalyzer.calculate_portfolio_metrics()`
- Call options analyzer if `--options` flag enabled
- Integrate options metrics into portfolio report

---

## Phase 6: LLM Integration

### 6.1 Update `llm_interface.py`
Add new prompt templates:

**`options_narrative_prompt`**
- Generates executive summary of options opportunities
- Explains detected strategies in plain language
- Risk/reward analysis for top 3 strategies

**`options_recommendations_prompt`**
- Personalized strategy suggestions based on:
  - Current portfolio holdings (covered call opportunities)
  - Market conditions (high IV → premium selling)
  - Risk tolerance (from user or inferred)

**`portfolio_hedging_prompt`**
- Portfolio-level hedging recommendations
- Greeks balancing suggestions
- Tail risk protection strategies

### 6.2 Update `generate_detailed_report()`
- Add options section if `options_analysis` present
- Include strategy analysis and recommendations
- Integrate options charts into report

---

## Phase 7: Orchestration

### 7.1 Update `orchestrator.py`

**Modify `__init__()`:**
- Add `self.options_analyzer = OptionsAnalyzer()` initialization
- Add `self.options_fetcher = OptionsDataFetcher()`

**Extend `analyze_ticker()`:**
- Add `include_options: bool` parameter
- If enabled:
  1. Fetch options chains via `options_fetcher`
  2. Calculate all Greeks for all contracts
  3. Detect strategies
  4. Generate P&L scenarios
  5. Attach `TickerOptionsAnalysis` to result

**Update `_analyze_all_tickers()`:**
- Pass `include_options` flag through to `analyze_ticker()`

**New method `_calculate_portfolio_options_metrics()`:**
- Aggregate Greeks across portfolio
- Generate hedging recommendations
- Attach to `PortfolioMetrics`

**Update `run()` method:**
- Check `request.include_options` flag
- Generate options-specific charts
- Pass options data to report generation

---

## Phase 8: CLI & Configuration

### 8.1 Update `config.py`
Add fields to `Config` dataclass:
- `include_options: bool = False` (default off)
- `options_cache_ttl_hours: int = 1` (shorter TTL for options)
- `greeks_method: str = "bsm"` (Black-Scholes-Merton or binomial)
- `strategy_detection_enabled: bool = True`

### 8.2 Update `main.py`
**Add CLI arguments:**
```python
parser.add_argument("--options", "--with-options",
                    action="store_true",
                    help="Include options analysis")
parser.add_argument("--options-expirations",
                    type=int, default=3,
                    help="Number of expirations to analyze")
```

**Update request creation:**
- Pass `include_options` to `PortfolioRequest`
- Validate options-specific parameters

### 8.3 Update `models.py` (request models)
- Add `include_options: bool = False` to `PortfolioRequest`
- Add `options_expirations: int = 3` to `PortfolioRequest`

---

## Phase 9: HTML Report Generation

### 9.1 Update `html_generator.py`
**Extend markdown parser to recognize:**
- Options chain tables
- Strategy cards with P&L diagrams
- Greeks summary tables
- Hedging recommendations section

**Add CSS styling for:**
- Options heatmaps (embedded as base64)
- Strategy comparison cards
- Greeks dashboard layout
- IV surface 3D charts

---

## Phase 10: Testing

### 10.1 Create `test_options_analyzer.py`
Unit tests for:
- Greeks calculation accuracy (compare to known values)
- IV solver convergence
- Strategy detection logic
- P&L calculation correctness

### 10.2 Create `test_options_fetcher.py`
- Cache validation for options data
- Error handling (no options available, expired contracts)
- Data parsing from yfinance

### 10.3 Create `test_options_integration.py`
- End-to-end test with real ticker (SPY)
- Verify all charts generated
- Validate report sections
- Portfolio-level aggregation

### 10.4 Update `test_orchestrator.py`
- Add test cases with `--options` flag
- Verify optional behavior (off by default)

---

## Implementation Order

1. **Phase 1**: Data models (1-2 hours)
2. **Phase 2**: Data fetching (2-3 hours)
3. **Phase 3**: Analysis engine - core Greeks (3-4 hours)
4. **Phase 3**: Analysis engine - strategy detection (3-4 hours)
5. **Phase 4**: Visualizations (4-5 hours)
6. **Phase 5**: Portfolio integration (2-3 hours)
7. **Phase 6**: LLM integration (2-3 hours)
8. **Phase 7**: Orchestration updates (2-3 hours)
9. **Phase 8**: CLI & config (1-2 hours)
10. **Phase 9**: HTML generation (2-3 hours)
11. **Phase 10**: Testing (3-4 hours)

**Total Estimated Time**: 25-36 hours

---

## Key Design Decisions

✅ **Optional by default** - No performance impact when not used
✅ **Follows existing patterns** - Reuses caching, threading, error handling
✅ **Portfolio-aware** - Not just per-ticker, but cross-portfolio analysis
✅ **LLM-enhanced** - AI-generated strategy recommendations and explanations
✅ **Comprehensive visualization** - All 4 requested chart types
✅ **Production-ready** - Full error handling, logging, testing

---

## Example Usage

```bash
# Basic report without options
python main.py --tickers AAPL,MSFT --period 1y

# With options analysis
python main.py --tickers AAPL,MSFT --period 1y --options

# Options analysis with custom expirations
python main.py --tickers SPY --period 6mo --options --options-expirations 5

# Natural language with options
python main.py --request "Analyze TSLA with options strategies" --options
```

---

## File Structure After Implementation

```
financial_reporting_agent/
├── models_options.py          # NEW: Options data models
├── options_fetcher.py          # NEW: Options data retrieval
├── options_analyzer.py         # NEW: Greeks, strategies, P&L
├── models.py                   # MODIFIED: Add options fields
├── constants.py                # MODIFIED: Add options parameters
├── analyzers.py                # MODIFIED: Portfolio options integration
├── charts.py                   # MODIFIED: Add 4 new chart types
├── llm_interface.py            # MODIFIED: Add options prompts
├── orchestrator.py             # MODIFIED: Orchestrate options flow
├── config.py                   # MODIFIED: Add options config
├── main.py                     # MODIFIED: Add CLI flags
├── html_generator.py           # MODIFIED: Options report sections
├── test_options_analyzer.py    # NEW: Options analysis tests
├── test_options_fetcher.py     # NEW: Options fetching tests
├── test_options_integration.py # NEW: End-to-end tests
└── test_orchestrator.py        # MODIFIED: Add options tests
```

---

## User Requirements (from consultation)

- **Scope**: Advanced - Full strategy analysis and recommendations
- **Activation**: Optional flag (--options or --with-options)
- **Visualizations**: All 4 types (heatmaps, Greeks, P&L, IV surface)
- **Integration**: Yes - Portfolio-level options metrics

---

## Dependencies

The implementation will use existing dependencies:
- `yfinance` - Options chain data retrieval
- `numpy` - Mathematical calculations for Greeks and P&L
- `scipy` - Optimization for IV calculation
- `pandas` - Data manipulation
- `matplotlib` - All visualizations
- `langchain` - LLM prompts for strategy recommendations

No new dependencies required.

---

## Risk Mitigation

1. **Data availability**: Not all tickers have options → Handle gracefully with error messages
2. **Calculation accuracy**: Greeks validation against known benchmark values in tests
3. **Performance**: Options analysis only runs when flag is set, minimal overhead
4. **Complexity**: Phased implementation allows incremental testing and validation
5. **Backwards compatibility**: All changes are additive, no breaking changes to existing functionality

---

## Success Criteria

- [ ] All 10 phases completed
- [ ] All tests passing (>95% coverage for new code)
- [ ] Report generation works with and without `--options` flag
- [ ] All 4 visualization types rendering correctly
- [ ] Portfolio-level Greeks aggregation accurate
- [ ] LLM generates coherent strategy recommendations
- [ ] HTML report includes options sections with proper styling
- [ ] Performance overhead <10% when options disabled
- [ ] Documentation updated (CLAUDE.md, README.md)

---

## Next Steps

1. Begin with Phase 1 (Data Models)
2. Implement each phase sequentially
3. Test thoroughly after each phase
4. Update documentation as features are added
5. Create example reports showcasing new capabilities
