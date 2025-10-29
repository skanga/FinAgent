# 📊 Advanced Financial Reporting Agent

> AI-powered financial analysis tool that generates comprehensive reports with technical indicators, fundamental analysis, and portfolio metrics.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

### Core Capabilities
- **📈 Technical Analysis**: RSI, MACD, Bollinger Bands, Stochastic, Moving Averages, Volatility
- **💰 Fundamental Analysis**: P/E ratios, revenue growth, earnings, cash flow, margins
- **📊 Portfolio Analytics**: Diversification ratio, correlation matrix, Sharpe ratio, max drawdown
- **🎯 Options Analysis**: Greeks (Delta/Gamma/Theta/Vega/Rho), IV analysis, strategy detection, P&L modeling
- **🤖 AI-Powered Insights**: LLM-generated narratives and investment recommendations
- **📉 Advanced Metrics**: Alpha, Beta, Sortino, Calmar, Treynor, Information Ratio
- **🔔 Smart Alerts**: Automated detection of overbought/oversold conditions
- **💾 Intelligent Caching**: Reduces API calls with configurable TTL (dual-format: Parquet + Pickle)
- **📝 Natural Language Requests**: "Analyze tech stocks over 6 months"
- **🌐 HTML Reports**: Professional, responsive HTML reports with embedded charts
- **🚀 Browser Launch**: Optional browser opening with user confirmation (security best practice)

### Performance & Quality
- **🚀 Optimized Operations**: Vectorized DataFrame operations for speed
- **🧪 Comprehensive Testing**: Memory leak prevention, performance benchmarks
- **🔒 Security First**: Path traversal protection, API key masking, no pickle vulnerabilities
- **📦 Clean Code**: Type hints, DRY principles, explicit error handling

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
  - [CLI Examples](#cli-examples)
  - [Natural Language Requests](#natural-language-requests)
  - [Advanced Options](#advanced-options)
- [Architecture](#-architecture)
- [Output](#-output)
- [API Providers](#-api-providers)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 📚 Additional Documentation

> **[📑 Documentation Index](docs/README.md)** - Complete navigation guide to all documentation

### Core Guides
- **[Architecture Guide](docs/ARCHITECTURE.md)** - Complete system design with diagrams, concurrency model, and connection pooling
- **[Configuration Examples](docs/CONFIG_EXAMPLES.md)** - LLM providers, performance tuning, and deployment (Docker, K8s, Lambda)
- **[Retry Behavior](docs/RETRY_BEHAVIOR.md)** - Network resilience, error handling, and troubleshooting

### Development
- **[Development Guide](CLAUDE.md)** - For contributors and Claude Code users
- **[Testing Guide](docs/TESTING_QUICK_START.md)** - How to run tests
- **[Feature Docs](docs/FEATURE_COMPARATIVE_ANALYSIS.md)** - Comparative analysis feature

---

## 🚀 Quick Start

Get up and running in 3 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/skanga/FinAgent
cd financial_reporting_agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Run your first analysis
python main.py --tickers AAPL,MSFT,GOOGL --period 1y

# 5. (Optional) Include options analysis
python main.py --tickers AAPL --period 1y --options
```

The application will prompt you to open the HTML report in your browser. Press Enter (or type 'y') to view the report with charts, metrics, and AI insights! Markdown and data files are also saved in `./financial_reports/`.

---

## 📦 Installation

### Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **pip** package manager
- **API Key** for OpenAI-compatible LLM (OpenAI, Groq, Azure, etc.)

### Standard Installation

```bash
# Install all required dependencies
pip install -r requirements.txt
```

### Development Installation

For contributing or testing:

```bash
# Install with development dependencies
pip install -r requirements.txt

# Optionally install type checking
pip install mypy

# Run tests (all tests are now in tests/ directory)
pytest tests/ -v
```

### Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| `pandas` | Data manipulation | ≥2.0.0 |
| `numpy` | Numerical operations | ≥1.24.0 |
| `yfinance` | Financial data fetching | ≥0.2.28 |
| `matplotlib` | Chart generation | ≥3.7.0 |
| `langchain-openai` | LLM integration | ≥0.1.0 |
| `python-dotenv` | Environment config | ≥1.0.0 |

---

## ⚙️ Configuration

> **📖 For extensive configuration examples and deployment scenarios, see [CONFIG_EXAMPLES.md](docs/CONFIG_EXAMPLES.md)**

### Environment Variables

Configuration is managed through environment variables, which can be set in a `.env` file or directly in your shell.

#### Creating .env File

```bash
# Copy the example file
cp .env.example .env

# Edit with your favorite editor
nano .env  # or vim, code, etc.
```

#### Required Variables

```bash
# OpenAI-compatible API key (required)
OPENAI_API_KEY=your_api_key_here
```

#### Optional Variables

```bash
# API endpoint (default: https://api.openai.com/v1)
OPENAI_BASE_URL=https://api.openai.com/v1

# Model name (default: gpt-4o)
OPENAI_MODEL=gpt-4o

# Parallel processing threads (default: 3, range: 1-10)
MAX_WORKERS=3

# Cache expiration in hours (default: 24, range: 1-168)
CACHE_TTL_HOURS=24

# Benchmark for comparison (default: SPY)
BENCHMARK_TICKER=SPY

# HTML report generation (default: true)
GENERATE_HTML=true

# Embed images as base64 in HTML (default: false)
EMBED_IMAGES_IN_HTML=false

# Prompt to open reports in browser (default: true, press Enter to accept)
OPEN_IN_BROWSER=true

# Options analysis (default: false)
INCLUDE_OPTIONS=false

# Options cache TTL in hours (default: 1)
OPTIONS_CACHE_TTL_HOURS=1

# Number of options expirations to analyze (default: 3, max: 10)
OPTIONS_EXPIRATIONS=3
```

#### Configuration Validation

The application validates all configuration on startup:

- API key must be present and non-empty
- Numeric values must be within acceptable ranges
- Invalid configuration will produce clear error messages

### Alternative: System Environment Variables

For CI/CD or containerized environments:

```bash
# Linux/macOS
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_BASE_URL="https://api.openai.com/v1"

# Windows CMD
set OPENAI_API_KEY=sk-...
set OPENAI_BASE_URL=https://api.openai.com/v1
```

---

## 💡 Usage

### CLI Examples

#### Basic Analysis

```bash
# Analyze single stock over 1 year
python main.py --tickers AAPL --period 1y

# Analyze multiple stocks
python main.py --tickers AAPL,MSFT,GOOGL --period 1y

# Shorter time periods
python main.py --tickers TSLA --period 3mo
python main.py --tickers NVDA --period 6mo
```

#### Custom Output Directory

```bash
# Specify output location
python main.py --tickers AAPL,MSFT --period 1y --output ./my_reports

# Output is validated for security (must be in cwd or home directory)
```

#### Portfolio Analysis with Weights

```bash
# Equal weights (default)
python main.py --tickers AAPL,MSFT,GOOGL --period 1y

# Custom portfolio weights (must sum to 1.0)
python main.py --tickers AAPL,MSFT,GOOGL --weights 0.5,0.3,0.2 --period 1y
```

#### Cache Management

```bash
# Clear expired cache before running
python main.py --tickers AAPL --period 1y --clear-cache

# Expired cache is automatically cleared on each run
```

#### Verbose Logging

```bash
# Enable debug logging for troubleshooting
python main.py --tickers AAPL --period 1y --verbose
```

#### HTML Report Options

```bash
# Disable HTML generation (markdown only)
python main.py --tickers AAPL --period 1y --no-html

# Don't open in browser automatically
python main.py --tickers AAPL --period 1y --no-browser

# Embed images as base64 (single-file HTML, larger size)
python main.py --tickers AAPL --period 1y --embed-images

# Combine options
python main.py --tickers AAPL,MSFT --period 6mo --embed-images --no-browser
```

#### Options Analysis

```bash
# Enable options analysis for a single ticker
python main.py --tickers AAPL --period 1y --options

# Analyze more expiration dates (default: 3, max: 10)
python main.py --tickers SPY --period 6mo --options --options-expirations 5

# Portfolio with options analysis
python main.py --tickers AAPL,MSFT,GOOGL --period 1y --options

# Combined with other flags
python main.py --tickers AAPL --period 3mo --options --embed-images --verbose
```

**What you get with --options:**
- **Greeks Analysis**: Delta, Gamma, Theta, Vega, Rho for all contracts
- **IV Analysis**: Implied volatility skew, term structure, IV vs HV comparison
- **Strategy Detection**: 18+ strategies (straddles, spreads, condors, butterflies, etc.)
- **P&L Modeling**: Breakeven points, max profit/loss, probability of profit
- **Portfolio Greeks**: Aggregate exposure across all positions
- **Visualizations**: Options chain heatmaps, Greeks charts, P&L diagrams, IV surfaces
- **AI Recommendations**: LLM-generated strategy suggestions and hedging advice

### Natural Language Requests

Use natural language to specify your analysis:

```bash
# Natural language request
python main.py --request "Compare AAPL and MSFT over the past year"

# More examples
python main.py --request "Analyze tech stocks over 6 months"
python main.py --request "Show me Tesla's performance this quarter"
python main.py --request "Compare semiconductor stocks year to date"
```

The LLM will parse your request and extract:
- Ticker symbols
- Time period
- Analysis type

### Advanced Options

#### Available Periods

| Period | Description |
|--------|-------------|
| `1d` | 1 day |
| `5d` | 5 days |
| `1mo` | 1 month |
| `3mo` | 3 months |
| `6mo` | 6 months |
| `1y` | 1 year (default) |
| `2y` | 2 years |
| `5y` | 5 years |
| `10y` | 10 years |
| `ytd` | Year to date |
| `max` | Maximum available |

#### Help and Documentation

```bash
# Show all available options
python main.py --help

# View examples
python main.py --help | grep -A 20 "Examples:"
```

---

## 🏗️ Architecture

> **📖 For detailed architecture documentation with diagrams, see [ARCHITECTURE.md](docs/ARCHITECTURE.md)**

### Project Structure

```
financial_reporting_agent/
├── config.py              # Configuration management with .env support
├── models.py              # Stock data models and type definitions
├── models_options.py      # Options data models (9 dataclasses)
├── constants.py           # Application constants and parameters
├── cache.py               # Dual-format caching (Parquet + Pickle)
├── fetcher.py             # yfinance stock data fetching
├── options_fetcher.py     # yfinance options data fetching
├── analyzers.py           # Financial metrics computation
│   ├── AdvancedFinancialAnalyzer
│   ├── PortfolioAnalyzer
│   └── PortfolioOptionsAnalyzer
├── options_analyzer.py    # Options Greeks, IV, and strategy detection
├── charts.py              # Matplotlib chart generation (stock + options)
├── llm_interface.py       # LLM integration for narratives
├── html_generator.py      # HTML report generation
├── alerts.py              # Alert detection system
├── utils.py               # Utility functions
├── orchestrator.py        # Main coordination logic
├── main.py                # CLI entry point
├── templates/             # HTML templates
│   └── report_template.html
├── tests/                 # Test suite (600+ tests)
│   ├── conftest.py        # Shared test fixtures
│   └── test_*.py          # Test modules
├── requirements.txt       # Python dependencies
├── requirements-dev.txt   # Development dependencies
├── pytest.ini             # Pytest configuration
├── .env.example           # Configuration template
├── .gitignore             # Git exclusions
└── README.md              # This file

Generated directories:
├── .cache/                # Cached data (Parquet for DataFrames, Pickle for objects)
└── financial_reports/     # Generated reports (HTML, MD) and charts
```

### Component Overview

#### 1. **Configuration (`config.py`)**
- Loads `.env` file automatically using python-dotenv
- Validates all settings on startup
- Masks API keys in logs for security
- Detects LLM provider from base URL

#### 2. **Data Fetching (`fetcher.py`)**
- Wraps yfinance API with error handling
- Integrates with cache layer
- Timeout protection (configurable)

#### 3. **Caching (`cache.py`)**
- **Dual-Format Support**:
  - Parquet for pandas DataFrames (stock prices, fundamentals)
  - Pickle for Python objects (options chains, expirations)
- **TTL**: Configurable expiration (stock: 24h default, options: 1h default)
- **Security**: Path traversal protection, MD5-based keys
- **Performance**: Snappy compression for Parquet
- **Storage**: `.cache/` directory

#### 4. **Financial Analysis (`analyzers.py`)**

**AdvancedFinancialAnalyzer**:
- Technical indicators (RSI, MACD, Bollinger Bands)
- Advanced metrics (Sharpe, Sortino, Calmar, Treynor)
- Fundamental data parsing (revenue, earnings, margins)
- Benchmark comparison (Alpha, Beta, Information Ratio)

**PortfolioAnalyzer**:
- Portfolio-level returns and volatility
- Correlation matrix
- Diversification ratio
- Top contributors analysis
- Concentration risk (Herfindahl index)

**PortfolioOptionsAnalyzer**:
- Aggregate portfolio Greeks (Delta, Vega, Theta)
- Cross-position hedging recommendations
- Greeks concentration risk
- Vega/Delta/Theta balancing suggestions

#### 4.5. **Options Analysis (`options_analyzer.py`, `options_fetcher.py`)**
- **Greeks Calculation**: Black-Scholes-Merton for Delta, Gamma, Theta, Vega, Rho
- **IV Solver**: Newton-Raphson with Brent's method fallback
- **Strategy Detection**: 18+ patterns (straddles, spreads, condors, butterflies, etc.)
- **P&L Modeling**: Breakeven calculation, max profit/loss, probability of profit
- **IV Analysis**: Skew detection, term structure, IV vs historical volatility
- **Caching**: TTL-based caching of options chains (1 hour default)

#### 5. **Visualization (`charts.py`)**
- **Stock Charts**: Price with technical overlays, comparison charts, risk-reward scatter
- **Options Charts**:
  - Options chain heatmaps (volume, OI, IV)
  - Greeks visualization (Delta, Gamma, Theta, Vega)
  - P&L diagrams with breakeven points
  - IV surface/skew plots (3D and 2D)
- **Thread-safe**: Agg backend for concurrent generation
- **Memory-optimized**: Explicit cleanup with figure closing

#### 6. **LLM Integration (`llm_interface.py`)**
- Natural language request parsing
- Comprehensive narrative generation (stock + options)
- **Options-specific narratives**:
  - IV assessment and opportunities
  - Strategy recommendations with rationale
  - Portfolio hedging analysis
- Report quality review and scoring
- Temperature-controlled for consistency
- Chart embedding in markdown reports

#### 6.5. **HTML Generation (`html_generator.py`)**
- Markdown-to-HTML conversion (no external dependencies)
- Professional, responsive template
- Chart embedding (relative paths or base64)
- Print-friendly styling

#### 7. **Orchestration (`orchestrator.py`)**
- Coordinates all components
- Parallel ticker analysis
- Progress tracking
- Error handling and partial results
- Performance metrics collection

### Data Flow

```
User Input (CLI)
    ↓
Natural Language Parser (optional)
    ↓
Orchestrator
    ↓
┌─────────────┬───────────────┬──────────────┐
│   Fetcher   │   Analyzer    │  LLM         │
│  (cached)   │  (vectorized) │  (insights)  │
└─────────────┴───────────────┴──────────────┘
    ↓               ↓               ↓
Cache (.parquet)  Charts (.png)  Report (.md)
    ↓                               ↓
    │                          HTML Generator
    │                               ↓
    └─────────────→  Output Directory (./financial_reports/)
                     ├── report.html ⭐ Prompt to open in browser
                     ├── report.md
                     ├── charts.png (embedded in reports)
                     └── data.csv
```

---

## 📄 Output

### Report Structure

Each analysis generates:

#### 1. **HTML Report** (`financial_report_YYYYMMDDTHHMMSSZ.html`)

**Primary output format** - prompts to open in your browser (press Enter):
- Professional, responsive design
- Embedded technical charts with analysis
- Styled tables and metrics
- Print-friendly layout
- Works on desktop, tablet, and mobile

#### 2. **Markdown Report** (`financial_report_YYYYMMDDTHHMMSSZ.md`)

**Source format** for the HTML report:

```markdown
# Financial Analysis Report
Generated: 2024-01-15 14:30:00 UTC

## Executive Summary
[AI-generated overview of findings]

## Technical Charts
### Portfolio Comparison
![Portfolio Comparison](comparison_chart.png)

### AAPL Technical Analysis
![AAPL Technical Chart](AAPL_technical.png)

## Individual Stock Analysis
### AAPL
- Latest Price: $185.50
- Sharpe Ratio: 1.45
- Max Drawdown: -12.5%

## Portfolio Analysis
- Total Value: $10,000
- Portfolio Return: 15.3%
- Sharpe Ratio: 1.67
- Diversification Ratio: 1.23

## Alerts
⚠️ AAPL: RSI indicates overbought (75.2)

## AI Insights
[Detailed analysis and recommendations]
```

#### 3. **Charts** (PNG format)

For each ticker:
- `{TICKER}_technical.png`: Price, moving averages, Bollinger Bands, RSI, Stochastic

For portfolio:
- `comparison_chart.png`: Normalized returns comparison

Charts are automatically embedded in both HTML and markdown reports.

#### 4. **Data Files** (CSV format)

For each ticker:
- `{TICKER}_prices.csv`: Full historical data with computed indicators

### Performance Metrics

Each report includes execution metrics:

```json
{
  "execution_time_seconds": 12.5,
  "tickers_analyzed": 3,
  "successful": 3,
  "failed": 0,
  "charts_generated": 4,
  "quality_score": 8.5,
  "portfolio_analyzed": true
}
```

### Example Output

After running:
```bash
python main.py --tickers AAPL,MSFT,GOOGL --period 1y
```

You'll get:
```
financial_reports/
├── financial_report_20240115T143000Z.html    ⭐ Prompt to open in browser
├── financial_report_20240115T143000Z.md
├── AAPL_technical.png                         📊 Embedded in reports
├── AAPL_prices.csv
├── MSFT_technical.png                         📊 Embedded in reports
├── MSFT_prices.csv
├── GOOGL_technical.png                        📊 Embedded in reports
├── GOOGL_prices.csv
└── comparison_chart.png                       📊 Embedded in reports
```

---

## 🔌 API Providers

The agent supports any OpenAI-compatible API endpoint.

### OpenAI (Default)

```bash
# .env configuration
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
```

**Get API Key**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### Groq (Fast & Free Tier)

```bash
# .env configuration
OPENAI_API_KEY=gsk_...
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.1-70b-versatile
```

**Available Models**:
- `llama-3.1-70b-versatile` (recommended)
- `llama-3.1-8b-instant` (faster, lower quality)
- `mixtral-8x7b-32768`

**Get API Key**: [console.groq.com](https://console.groq.com)

### Azure OpenAI

```bash
# .env configuration
OPENAI_API_KEY=your_azure_key
OPENAI_BASE_URL=https://your-resource.openai.azure.com/
OPENAI_MODEL=gpt-4
```

**Setup**: [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)

### Local LLMs (Ollama, LM Studio, etc.)

```bash
# .env configuration
OPENAI_API_KEY=not-needed
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama2
```

**Compatible with**:
- Ollama
- LM Studio
- LocalAI
- vLLM
- Text Generation WebUI (with OpenAI extension)

### Provider Comparison

| Provider | Speed | Cost | Quality | Free Tier |
|----------|-------|------|---------|-----------|
| OpenAI GPT-4 | Medium | High | Excellent | No |
| OpenAI GPT-4o | Fast | Medium | Excellent | No |
| Groq (Llama 3.1) | Very Fast | Free/Low | Good | Yes |
| Azure OpenAI | Medium | High | Excellent | No |
| Local (Ollama) | Fast | Free | Varies | Yes |

---

## 🛠️ Development

### Running Tests

The project includes 600+ comprehensive tests in the `tests/` directory:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_memory_leaks.py -v

# Run tests by category
pytest tests/ -m unit              # Fast unit tests
pytest tests/ -m integration       # Integration tests
pytest tests/ -m "not slow"        # Skip slow tests

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# See TESTING_QUICK_START.md for more options
```

### Code Quality

**Optimizations implemented**:
- ✅ Vectorized DataFrame operations
- ✅ Single iloc[-1] extraction (1.44x speedup)
- ✅ Pre-calculated window sizes
- ✅ Explicit memory cleanup with gc.collect()
- ✅ Helper methods to eliminate duplication

**Security measures**:
- ✅ Path traversal protection
- ✅ No pickle serialization (uses Parquet)
- ✅ API key masking in logs
- ✅ Input validation for tickers and weights
- ✅ Specific exception handling (no broad catches)

**Code standards**:
- Type hints throughout
- Comprehensive docstrings
- DRY principle applied
- Explicit error messages
- No silent failures

### Performance Benchmarks

From `test_performance.py`:

```
DataFrame Size    Processing Time
500 rows          2.74 ms
1000 rows         2.69 ms
2500 rows         3.60 ms
5000 rows         4.18 ms

Scaling: 1.5x for 10x data (excellent)
iloc optimization: 1.44x speedup
```

### Memory Benchmarks

From `test_memory_leaks.py`:

```
5 large charts: 0.47 MB increase
5 comparison charts: 0.44 MB increase

All figures properly closed ✓
Explicit garbage collection ✓
```

---

## 🐛 Troubleshooting

> **📖 For detailed retry behavior and network error handling, see [RETRY_BEHAVIOR.md](docs/RETRY_BEHAVIOR.md)**

### Common Issues

#### 1. API Key Not Found

**Error**: `ValueError: OPENAI_API_KEY environment variable must be set`

**Solutions**:
```bash
# Check if .env file exists
ls -la .env

# Verify .env format (no 'set' prefix)
cat .env

# Create from example
cp .env.example .env
nano .env
```

#### 2. Module Import Errors

**Error**: `ModuleNotFoundError: No module named 'dotenv'`

**Solution**:
```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep dotenv
```

#### 3. Ticker Not Found

**Error**: `Failed to analyze INVALID: ValueError: Failed to fetch`

**Solutions**:
- Verify ticker symbol is correct (use Yahoo Finance format)
- Check internet connectivity
- Some tickers may require exchange suffix (e.g., `TSLA.L` for London)

#### 4. Cache Permission Errors

**Error**: `OSError: [Errno 13] Permission denied: '.cache'`

**Solution**:
```bash
# Remove cache directory and recreate
rm -rf .cache
mkdir .cache

# Or clear cache on next run
python main.py --tickers AAPL --clear-cache
```

#### 5. LLM Timeout

**Error**: `Request timeout after 30 seconds`

**Solutions**:
- Check API endpoint is accessible
- Verify API key is valid
- Try a different model (smaller/faster)
- Check your internet connection

#### 6. Memory Issues

**Error**: `MemoryError: Unable to allocate array`

**Solutions**:
```bash
# Reduce number of tickers
python main.py --tickers AAPL,MSFT  # instead of 20 tickers

# Use shorter period
python main.py --tickers AAPL --period 3mo  # instead of 10y

# Reduce worker threads
export MAX_WORKERS=1
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# CLI flag
python main.py --tickers AAPL --verbose

# Check logs for details
# Logs include: API calls, cache hits/misses, timing, errors
```

### Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Review the error message carefully
3. Enable `--verbose` mode for detailed logs
4. Check the [Issues](issues) section
5. Verify your configuration with:
   ```bash
   pytest tests/test_env_loading.py -v
   ```

---

## 🤝 Contributing

Contributions are welcome! This project values:

- Clean, maintainable code
- Comprehensive testing
- Security-first approach
- Clear documentation

### Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/skanga/financial_reporting_agent.git
cd financial_reporting_agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env for testing
cp .env.example .env
# Add your API key

# 5. Run tests
pytest tests/ -v
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions
- Add tests for new features
- Keep functions focused and small
- Use meaningful variable names

### Pull Request Process

1. Create a feature branch: `git checkout -b feature/amazing-feature`
2. Make your changes with clear commits
3. Run all tests and ensure they pass
4. Update documentation as needed
5. Submit PR with description of changes

### Areas for Contribution

- Additional technical indicators
- More chart types
- Additional LLM providers
- Performance optimizations
- Documentation improvements
- Bug fixes

---

## 📜 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **yfinance**: Market data retrieval
- **pandas/numpy**: Data processing
- **matplotlib**: Visualization
- **LangChain**: LLM integration
- **python-dotenv**: Configuration management

---

## 📞 Support

- **Documentation**: This README and inline code comments
- **Issues**: [GitHub Issues](issues)
- **Questions**: Open a discussion or issue

---

## 🗺️ Roadmap

### Recently Completed ✅
- [x] HTML report generation with professional styling
- [x] Chart embedding in reports (markdown + HTML)
- [x] Auto-launch reports in browser
- [x] Responsive design for mobile/tablet/desktop
- [x] Options Analytics (Greeks, IV, strategies, P&L modeling, portfolio hedging)
- [x] Dual-format caching (Parquet + Pickle)
- [x] Stochastic oscillator indicator
- [x] LLM quality review system

### Future Enhancements

- [ ] Additional Data Sources (not only yfinance)
- [ ] Real-time data streaming
- [ ] Advanced portfolio optimization (Markowitz, Black-Litterman)
- [ ] Sector and industry analysis
- [ ] Backtesting framework
- [ ] Web interface (Flask/FastAPI)
- [ ] Database persistence (PostgreSQL/SQLite)
- [ ] Additional chart types (candlesticks, volume analysis)
- [ ] Interactive charts (Plotly integration)
- [ ] News sentiment integration
- [ ] PDF report generation
- [ ] Multi-currency support
- [ ] ESG metrics integration
- [ ] Dark mode for HTML reports
- [ ] More options strategies (ratio spreads, box spreads, synthetic positions)
- [ ] Historical IV percentile tracking
- [ ] Earnings calendar integration for options
- [ ] American options pricing (binomial trees)

---

**Built with ❤️ in the SF Bay Area**
