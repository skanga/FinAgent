# Configuration Examples

## Overview

The Financial Reporting Agent supports extensive customization through environment variables and programmatic configuration. This guide provides practical examples for common configuration scenarios.

## Table of Contents

1. [Basic Setup](#basic-setup)
2. [LLM Provider Configuration](#llm-provider-configuration)
3. [Performance Tuning](#performance-tuning)
4. [Cache Configuration](#cache-configuration)
5. [Custom Analysis Parameters](#custom-analysis-parameters)
6. [Programmatic Configuration](#programmatic-configuration)
7. [Production Deployment](#production-deployment)

---

## Basic Setup

### Minimal Configuration

Create a `.env` file in the project directory:

```bash
# .env - Minimal configuration
OPENAI_API_KEY=sk-proj-your-api-key-here
```

That's it! The system uses sensible defaults for everything else:
- Model: `gpt-4o`
- Max Workers: `3`
- Cache TTL: `24` hours
- Benchmark: `SPY`

### Recommended Configuration

```bash
# .env - Recommended for production
OPENAI_API_KEY=sk-proj-your-api-key-here
OPENAI_MODEL=gpt-4o
MAX_WORKERS=3
CACHE_TTL_HOURS=24
REQUEST_TIMEOUT=30
BENCHMARK_TICKER=SPY
RISK_FREE_RATE=0.02
```

---

## LLM Provider Configuration

### OpenAI (Default)

```bash
# .env - OpenAI configuration
OPENAI_API_KEY=sk-proj-your-openai-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# Alternative models:
# OPENAI_MODEL=gpt-4-turbo
# OPENAI_MODEL=gpt-3.5-turbo  # Faster, cheaper, less accurate
```

**Usage:**
```bash
python main.py --tickers AAPL,MSFT --period 1y
```

### Anthropic Claude (via OpenAI-compatible API)

```bash
# .env - Anthropic configuration
OPENAI_API_KEY=sk-ant-your-anthropic-key
OPENAI_BASE_URL=https://api.anthropic.com/v1
OPENAI_MODEL=claude-3-opus-20240229

# Alternative models:
# OPENAI_MODEL=claude-3-sonnet-20240229  # Balanced
# OPENAI_MODEL=claude-3-haiku-20240307   # Fast, cheaper
```

**Note:** Requires Anthropic's OpenAI-compatible endpoint.

### Google Gemini

```bash
# .env - Google Gemini configuration
OPENAI_API_KEY=your-google-api-key
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_MODEL=gemini-pro
```

### Ollama (Local Models)

Run local models with Ollama:

```bash
# First, start Ollama server
ollama serve

# Pull a model
ollama pull llama2

# .env - Ollama configuration
OPENAI_API_KEY=ollama  # Can be any value
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama2

# Alternative models:
# OPENAI_MODEL=mistral
# OPENAI_MODEL=codellama
# OPENAI_MODEL=neural-chat
```

**Benefits:**
- No API costs
- Complete data privacy
- No rate limits
- Works offline

**Tradeoffs:**
- Slower than cloud models
- May produce less accurate analysis
- Requires local GPU for best performance

### Custom OpenAI-Compatible Endpoints

Many services provide OpenAI-compatible APIs:

```bash
# .env - Custom endpoint
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-custom-endpoint.com/v1
OPENAI_MODEL=your-model-name
```

**Examples:**
- **LocalAI:** Self-hosted OpenAI alternative
- **Text Generation Inference:** Hugging Face's inference server
- **vLLM:** Fast LLM inference server

---

## Performance Tuning

### High-Performance Configuration

For faster analysis with multiple tickers:

```bash
# .env - High performance
MAX_WORKERS=5              # More concurrent analyses
REQUEST_TIMEOUT=15         # Fail faster on slow requests
CACHE_TTL_HOURS=48         # Longer cache (less API calls)
```

**Best for:**
- Large portfolios (10+ tickers)
- Fast internet connection
- Powerful multi-core CPU

**Example:**
```bash
# Analyze 10 stocks concurrently
python main.py --tickers AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,NFLX,AMD,INTC --period 1y
```

### Low-Resource Configuration

For constrained environments or slow connections:

```bash
# .env - Low resource
MAX_WORKERS=1              # Sequential processing
REQUEST_TIMEOUT=60         # More patient with slow responses
CACHE_TTL_HOURS=72         # Very long cache to minimize API calls
```

**Best for:**
- Limited CPU/memory
- Slow internet connection
- Rate-limited API keys
- Debugging

### Rate-Limited API Configuration

If you're hitting rate limits:

```bash
# .env - Respect rate limits
MAX_WORKERS=2              # Reduce concurrency
REQUEST_TIMEOUT=45         # Allow more time per request
```

**Additional strategies:**
```python
# In main.py, add delay between tickers
import time

for ticker in tickers:
    orchestrator.analyze_ticker(ticker, period, output_dir)
    time.sleep(1)  # 1 second delay between requests
```

---

## Cache Configuration

### Development: No Cache

Disable caching to always get fresh data:

```bash
# .env - No cache
CACHE_TTL_HOURS=0          # Disable cache
```

Or clear cache before each run:
```bash
python main.py --clear-cache --tickers AAPL --period 1y
```

### Production: Aggressive Caching

Minimize API calls with long cache:

```bash
# .env - Aggressive caching
CACHE_TTL_HOURS=168        # 1 week cache
```

**Use case:** Historical analysis where data doesn't change often.

### Custom Cache Directory

Change where cached data is stored:

```python
# custom_config.py
from config import Config
from cache import CacheManager
from pathlib import Path

config = Config.from_env()
cache = CacheManager(
    cache_dir=Path("/data/financial_cache"),
    cache_ttl_hours=config.cache_ttl_hours
)
```

### Cache Inspection

Check what's cached:

```bash
# List cached files
ls -lh .cache/

# Check cache size
du -sh .cache/

# Clear old cache (manually)
find .cache/ -name "*.parquet" -mtime +7 -delete
```

---

## Custom Analysis Parameters

### Change Benchmark

Use different benchmark index:

```bash
# .env - Use different benchmarks
BENCHMARK_TICKER=QQQ       # Nasdaq-100 instead of S&P 500
# BENCHMARK_TICKER=^DJI    # Dow Jones
# BENCHMARK_TICKER=^IXIC   # Nasdaq Composite
# BENCHMARK_TICKER=VTI     # Total Stock Market
```

### Change Risk-Free Rate

Adjust for current interest rates:

```bash
# .env - 2024 rates
RISK_FREE_RATE=0.05        # 5% (current Fed rate)

# Historical rates:
# RISK_FREE_RATE=0.00      # 2020-2021 (near zero)
# RISK_FREE_RATE=0.02      # 2019 (default)
```

**Impact:** Affects Sharpe ratio, Treynor ratio, and alpha calculations.

### Custom Analysis Periods

```bash
# Short-term analysis
python main.py --tickers AAPL --period 1mo

# Long-term analysis
python main.py --tickers AAPL --period 10y

# Year-to-date
python main.py --tickers AAPL --period ytd

# All available data
python main.py --tickers AAPL --period max
```

**Valid periods:** `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

---

## Programmatic Configuration

### Custom Config Class

Create a config programmatically:

```python
# my_analysis.py
from config import Config
from orchestrator import FinancialReportOrchestrator
from models import PortfolioRequest

# Create custom config
config = Config(
    openai_api_key="sk-proj-your-key",
    openai_base_url="https://api.openai.com/v1",
    model_name="gpt-4o",
    max_workers=5,
    cache_ttl_hours=48,
    request_timeout=30,
    risk_free_rate=0.05,
    benchmark_ticker="QQQ",
)

# Create orchestrator
orchestrator = FinancialReportOrchestrator(config)

# Run analysis
request = PortfolioRequest(
    tickers=["AAPL", "MSFT", "GOOGL"],
    period="1y",
    weights={"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}
)

result = orchestrator.run(request, output_dir="./my_reports")
print(f"Report saved to: {result.final_markdown_path}")
```

### Override Specific Settings

Mix environment and programmatic config:

```python
# Uses .env for most settings, override specific ones
config = Config.from_env()
config.max_workers = 10  # Override for this run
config.benchmark_ticker = "QQQ"  # Use Nasdaq benchmark

orchestrator = FinancialReportOrchestrator(config)
```

### Batch Analysis with Different Configs

Analyze different portfolios with different settings:

```python
# batch_analysis.py
from config import Config
from orchestrator import FinancialReportOrchestrator
from models import PortfolioRequest

# Conservative portfolio
conservative_config = Config(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o",
    risk_free_rate=0.05,
    benchmark_ticker="AGG",  # Bond index
)

# Aggressive portfolio
aggressive_config = Config(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o",
    risk_free_rate=0.05,
    benchmark_ticker="QQQ",  # Tech-heavy
)

# Run both analyses
conservative = FinancialReportOrchestrator(conservative_config)
conservative.run(
    PortfolioRequest(tickers=["BND", "AGG", "TLT"], period="1y"),
    output_dir="./reports/conservative"
)

aggressive = FinancialReportOrchestrator(aggressive_config)
aggressive.run(
    PortfolioRequest(tickers=["TQQQ", "SOXL", "ARKK"], period="1y"),
    output_dir="./reports/aggressive"
)
```

### Dynamic Config from User Input

```python
# interactive.py
from config import Config
from orchestrator import FinancialReportOrchestrator
from models import PortfolioRequest

def get_user_config():
    """Interactively build config from user input."""
    print("Configuration Builder")
    print("=" * 40)

    api_key = input("OpenAI API Key: ")
    model = input("Model (default: gpt-4o): ") or "gpt-4o"
    workers = int(input("Max Workers (default: 3): ") or "3")
    benchmark = input("Benchmark Ticker (default: SPY): ") or "SPY"

    return Config(
        openai_api_key=api_key,
        model_name=model,
        max_workers=workers,
        benchmark_ticker=benchmark,
    )

# Use it
config = get_user_config()
orchestrator = FinancialReportOrchestrator(config)

# Get tickers from user
tickers_input = input("Tickers (comma-separated): ")
tickers = [t.strip() for t in tickers_input.split(",")]

request = PortfolioRequest(tickers=tickers, period="1y")
result = orchestrator.run(request)
```

---

## Production Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Environment variables (override with docker run -e)
ENV OPENAI_API_KEY=""
ENV MAX_WORKERS=3
ENV CACHE_TTL_HOURS=24
ENV REQUEST_TIMEOUT=30

# Run
CMD ["python", "main.py", "--tickers", "AAPL,MSFT,GOOGL", "--period", "1y"]
```

Run with Docker:
```bash
# Build
docker build -t financial-agent .

# Run with environment variables
docker run -e OPENAI_API_KEY=sk-proj-your-key \
           -v $(pwd)/reports:/app/financial_reports \
           financial-agent

# Run with custom command
docker run -e OPENAI_API_KEY=sk-proj-your-key \
           financial-agent \
           python main.py --request "Analyze tech stocks AAPL and MSFT"
```

### Kubernetes ConfigMap

```yaml
# config-map.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: financial-agent-config
data:
  MAX_WORKERS: "5"
  CACHE_TTL_HOURS: "48"
  REQUEST_TIMEOUT: "30"
  BENCHMARK_TICKER: "SPY"
  RISK_FREE_RATE: "0.05"
  OPENAI_MODEL: "gpt-4o"
  OPENAI_BASE_URL: "https://api.openai.com/v1"

---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: financial-agent-secret
type: Opaque
stringData:
  OPENAI_API_KEY: sk-proj-your-api-key-here

---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: financial-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: financial-agent
  template:
    metadata:
      labels:
        app: financial-agent
    spec:
      containers:
      - name: financial-agent
        image: financial-agent:latest
        envFrom:
        - configMapRef:
            name: financial-agent-config
        - secretRef:
            name: financial-agent-secret
        volumeMounts:
        - name: cache
          mountPath: /app/.cache
        - name: reports
          mountPath: /app/financial_reports
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: financial-agent-cache
      - name: reports
        persistentVolumeClaim:
          claimName: financial-agent-reports
```

### AWS Lambda Configuration

```python
# lambda_handler.py
import json
import os
from config import Config
from orchestrator import FinancialReportOrchestrator
from models import PortfolioRequest

def lambda_handler(event, context):
    """AWS Lambda handler for financial analysis."""

    # Load config from environment variables
    config = Config(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model_name=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        max_workers=int(os.environ.get("MAX_WORKERS", "3")),
        cache_ttl_hours=int(os.environ.get("CACHE_TTL_HOURS", "24")),
    )

    # Parse request
    tickers = event.get("tickers", ["AAPL"])
    period = event.get("period", "1y")
    weights = event.get("weights")

    # Run analysis
    orchestrator = FinancialReportOrchestrator(config)
    request = PortfolioRequest(
        tickers=tickers,
        period=period,
        weights=weights
    )

    result = orchestrator.run(request, output_dir="/tmp/reports")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "report_path": str(result.final_markdown_path),
            "quality_score": result.performance_metrics.get("quality_score", 0),
            "tickers_analyzed": list(result.analyses.keys()),
        })
    }
```

Lambda environment variables:
```bash
OPENAI_API_KEY=sk-proj-your-key
OPENAI_MODEL=gpt-4o
MAX_WORKERS=2  # Lambda has limited CPU
CACHE_TTL_HOURS=168  # Long cache to reduce API calls
REQUEST_TIMEOUT=25  # Lambda 30s timeout - leave buffer
```

### Environment-Specific Configs

```bash
# .env.development
OPENAI_API_KEY=sk-proj-dev-key
OPENAI_MODEL=gpt-3.5-turbo  # Cheaper for testing
MAX_WORKERS=1
CACHE_TTL_HOURS=1  # Short cache for testing
REQUEST_TIMEOUT=10

# .env.staging
OPENAI_API_KEY=sk-proj-staging-key
OPENAI_MODEL=gpt-4o
MAX_WORKERS=3
CACHE_TTL_HOURS=12
REQUEST_TIMEOUT=30

# .env.production
OPENAI_API_KEY=sk-proj-prod-key
OPENAI_MODEL=gpt-4o
MAX_WORKERS=5
CACHE_TTL_HOURS=24
REQUEST_TIMEOUT=30
```

Load environment-specific config:
```bash
# Development
ln -sf .env.development .env
python main.py --tickers AAPL --period 1mo

# Production
ln -sf .env.production .env
python main.py --tickers AAPL,MSFT,GOOGL --period 1y
```

---

## Configuration Validation

### Check Current Configuration

```python
# check_config.py
from config import Config

config = Config.from_env()

print("Current Configuration:")
print("=" * 50)
print(f"Provider: {config.provider}")
print(f"Model: {config.model_name}")
print(f"Base URL: {config.openai_base_url}")
print(f"API Key: {config._mask_api_key(config.openai_api_key)}")
print(f"Max Workers: {config.max_workers}")
print(f"Cache TTL: {config.cache_ttl_hours} hours")
print(f"Request Timeout: {config.request_timeout}s")
print(f"Risk-Free Rate: {config.risk_free_rate * 100}%")
print(f"Benchmark: {config.benchmark_ticker}")
```

Output:
```
Current Configuration:
==================================================
Provider: openai
Model: gpt-4o
Base URL: https://api.openai.com/v1
API Key: sk-p...key_
Max Workers: 3
Cache TTL: 24 hours
Request Timeout: 30s
Risk-Free Rate: 2.0%
Benchmark: SPY
```

### Validate Before Running

```python
# validate_and_run.py
from config import Config

try:
    config = Config.from_env()
    print(f"✓ Configuration valid")
    print(f"✓ Provider: {config.provider}")
    print(f"✓ Model: {config.model_name}")

    # Test API key is set
    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY not set")

    print("✓ API key configured")
    print("\nStarting analysis...")

except ValueError as e:
    print(f"✗ Configuration error: {e}")
    print("\nPlease check your .env file")
    exit(1)
```

---

## Related Documentation

- [Architecture](ARCHITECTURE.md) - System design overview
- [Retry Behavior](RETRY_BEHAVIOR.md) - Network resilience
- [CLAUDE.md](CLAUDE.md) - Development guide
- [README.md](README.md) - Getting started
