# 📊 Advanced Financial Reporting Agent

> AI-powered financial analysis tool that generates comprehensive reports with technical indicators, fundamental analysis, and portfolio metrics.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

### Core Capabilities
- **📈 Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages, Volatility
- **💰 Fundamental Analysis**: P/E ratios, revenue growth, earnings, cash flow, margins
- **📊 Portfolio Analytics**: Diversification ratio, correlation matrix, Sharpe ratio, max drawdown
- **🤖 AI-Powered Insights**: LLM-generated narratives and investment recommendations
- **📉 Advanced Metrics**: Alpha, Beta, Sortino, Calmar, Treynor, Information Ratio
- **🔔 Smart Alerts**: Automated detection of overbought/oversold conditions
- **💾 Intelligent Caching**: Reduces API calls with configurable TTL
- **📝 Natural Language Requests**: "Analyze tech stocks over 6 months"

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

> **[📑 Documentation Index](DOCUMENTATION_INDEX.md)** - Complete navigation guide to all documentation

### Core Guides
- **[Architecture Guide](ARCHITECTURE.md)** - Complete system design with diagrams, concurrency model, and connection pooling
- **[Configuration Examples](CONFIG_EXAMPLES.md)** - LLM providers, performance tuning, and deployment (Docker, K8s, Lambda)
- **[Retry Behavior](RETRY_BEHAVIOR.md)** - Network resilience, error handling, and troubleshooting

### Development
- **[Development Guide](CLAUDE.md)** - For contributors and Claude Code users
- **[Testing Guide](TESTING_QUICK_START.md)** - How to run tests
- **[Feature Docs](FEATURE_COMPARATIVE_ANALYSIS.md)** - Comparative analysis feature

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
```

Your report will be generated in `./financial_reports/` with charts, metrics, and AI insights!

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

# Run tests
python test_memory_leaks.py
python test_performance.py
python test_yoy_refactor.py
python test_env_loading.py
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

> **📖 For extensive configuration examples and deployment scenarios, see [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md)**

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

> **📖 For detailed architecture documentation with diagrams, see [ARCHITECTURE.md](ARCHITECTURE.md)**

### Project Structure

```
financial_reporting_agent/
├── config.py              # Configuration management with .env support
├── models.py              # Data models and type definitions
├── cache.py               # Parquet-based caching system
├── fetcher.py             # yfinance data fetching wrapper
├── analyzers.py           # Financial metrics computation
│   ├── AdvancedFinancialAnalyzer
│   └── PortfolioAnalyzer
├── charts.py              # Matplotlib chart generation
├── llm_interface.py       # LLM integration for narratives
├── alerts.py              # Alert detection system
├── utils.py               # Utility functions
├── orchestrator.py        # Main coordination logic
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
├── .env.example           # Configuration template
├── .gitignore             # Git exclusions
└── README.md              # This file

Generated directories:
├── .cache/                # Cached price data (Parquet format)
└── financial_reports/     # Generated reports and charts
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
- **Format**: Apache Parquet (secure, efficient)
- **TTL**: Configurable expiration (1-168 hours)
- **Security**: Path traversal protection
- **Performance**: Compression with Snappy
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

#### 5. **Visualization (`charts.py`)**
- Price charts with technical overlays
- Comparison charts (normalized returns)
- Risk-reward scatter plots
- Memory-optimized with explicit cleanup

#### 6. **LLM Integration (`llm_interface.py`)**
- Natural language request parsing
- Comprehensive narrative generation
- Report quality review and scoring
- Temperature-controlled for consistency

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
    ↓
Output Directory (./financial_reports/)
```

---

## 📄 Output

### Report Structure

Each analysis generates:

#### 1. **Markdown Report** (`financial_report_YYYYMMDDTHHMMSSZ.md`)

```markdown
# Financial Analysis Report
Generated: 2024-01-15 14:30:00 UTC

## Executive Summary
[AI-generated overview of findings]

## Individual Stock Analysis
### AAPL
- Latest Price: $185.50
- Sharpe Ratio: 1.45
- Max Drawdown: -12.5%
- [Charts and metrics]

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

#### 2. **Charts** (PNG format)

For each ticker:
- `{TICKER}_technical.png`: Price, moving averages, Bollinger Bands, RSI

For portfolio:
- `comparison_chart.png`: Normalized returns comparison

#### 3. **Data Files** (CSV format)

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
├── financial_report_20240115T143000Z.md
├── AAPL_technical.png
├── AAPL_prices.csv
├── MSFT_technical.png
├── MSFT_prices.csv
├── GOOGL_technical.png
├── GOOGL_prices.csv
└── comparison_chart.png
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

The project includes comprehensive test suites:

```bash
# Test memory management
python test_memory_leaks.py

# Test performance optimizations
python test_performance.py

# Test code refactoring
python test_yoy_refactor.py

# Test .env file loading
python test_env_loading.py
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

> **📖 For detailed retry behavior and network error handling, see [RETRY_BEHAVIOR.md](RETRY_BEHAVIOR.md)**

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
   python test_env_loading.py
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
git clone https://github.com/yourusername/financial_reporting_agent.git
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
python test_memory_leaks.py
python test_performance.py
python test_yoy_refactor.py
python test_env_loading.py
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

Future enhancements planned:

- [ ] Real-time data streaming
- [ ] Advanced portfolio optimization (Markowitz, Black-Litterman)
- [ ] Sector and industry analysis
- [ ] Backtesting framework
- [ ] Web interface (Flask/FastAPI)
- [ ] Database persistence (PostgreSQL/SQLite)
- [ ] Additional chart types (candlesticks, volume analysis)
- [ ] Options analytics
- [ ] News sentiment integration
- [ ] PDF report generation
- [ ] Multi-currency support
- [ ] ESG metrics integration

---

**Built with ❤️ in the SF Bay Area**
