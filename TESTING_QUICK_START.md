# Testing Quick Start Guide

## TL;DR - Get Started in 15 Minutes

### 1. Install pytest (2 minutes)

```bash
pip install -r requirements-dev.txt
```

### 2. Run existing tests with pytest (1 minute)

```bash
# Run all tests
pytest

# Run specific test file
pytest test_pydantic_validation.py -v

# Run with summary
pytest --tb=short
```

### 3. Check test coverage (2 minutes)

```bash
# Install coverage plugin
pip install pytest-cov

# Run tests with coverage
pytest --cov=. --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
xdg-open htmlcov/index.html  # Linux
```

### 4. Run only fast tests (1 minute)

```bash
# Run only unit tests (fast)
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Skip network and LLM tests
pytest -m "not (network or llm)"
```

### 5. Run tests in parallel (1 minute)

```bash
# Install xdist plugin
pip install pytest-xdist

# Run tests in parallel (faster)
pytest -n auto
```

## Current Test Commands

### Run All Tests
```bash
# Verbose output
pytest -v

# Show print statements
pytest -s

# Show detailed failure info
pytest -vv
```

### Run Specific Tests
```bash
# Single file
pytest test_pydantic_validation.py

# Single test function
pytest test_pydantic_validation.py::test_ticker_request_validation

# Tests matching pattern
pytest -k "validation"

# Tests in directory
pytest tests/unit/
```

### Run by Category
```bash
# Only unit tests
pytest -m unit

# Only integration tests
pytest -m integration

# Slow tests only
pytest -m slow

# Everything except network tests
pytest -m "not network"

# Unit and integration, but not slow
pytest -m "unit or integration" -m "not slow"
```

## Test Output Examples

### Success
```
============================= test session starts ==============================
collected 42 items

test_pydantic_validation.py ✓✓✓✓✓✓✓✓✓✓                                  [ 23%]
test_retry_logic.py ✓✓✓                                                  [ 30%]
test_concurrent_processing.py ✓✓✓✓                                       [ 40%]

============================= 42 passed in 2.54s ===============================
```

### Failure
```
________________________________ test_cache_expiration _________________________

    def test_cache_expiration():
        cache = CacheManager(ttl_hours=0.001)
>       assert cache.get("AAPL", "1y", "prices") is None
E       AssertionError: assert <DataFrame> is not None

test_cache.py:15: AssertionError
========================== 1 failed, 41 passed in 2.67s ========================
```

## Coverage Report

### Command Line
```bash
pytest --cov=. --cov-report=term-missing

----------- coverage: platform win32, python 3.11.5 -----------
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
cache.py                   45     12    73%   34-38, 67-71
alerts.py                  28      8    71%   45-52
analyzers.py              156     45    71%   89-95, 123-145
...
TOTAL                    1234    345    72%
```

### HTML Report
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html

# Interactive HTML report showing:
# - Overall coverage %
# - Per-file coverage
# - Line-by-line highlighting (green = covered, red = not covered)
```

## Writing Your First Test

### 1. Create test file
```python
# tests/unit/test_cache.py
import pytest
from cache import CacheManager

def test_cache_stores_and_retrieves_data(temp_cache_dir, sample_price_data):
    """Test that cache can store and retrieve DataFrames."""
    # Arrange
    cache = CacheManager(cache_dir=temp_cache_dir, ttl_hours=24)

    # Act
    cache.set("AAPL", "1y", sample_price_data, "prices")
    result = cache.get("AAPL", "1y", "prices")

    # Assert
    assert result is not None
    assert len(result) == len(sample_price_data)
    assert result.equals(sample_price_data)
```

### 2. Run your test
```bash
pytest tests/unit/test_cache.py::test_cache_stores_and_retrieves_data -v
```

### 3. Check coverage
```bash
pytest tests/unit/test_cache.py --cov=cache --cov-report=term-missing
```

## Common pytest Commands Reference

### Discovery
```bash
# List all tests (don't run)
pytest --collect-only

# List tests matching pattern
pytest --collect-only -k "cache"

# Show available markers
pytest --markers
```

### Output Control
```bash
# Verbose
pytest -v

# Very verbose (show all details)
pytest -vv

# Quiet (minimal output)
pytest -q

# Show print statements
pytest -s

# Show local variables on failure
pytest -l
```

### Failure Handling
```bash
# Stop on first failure
pytest -x

# Stop after N failures
pytest --maxfail=3

# Run last failed tests
pytest --lf

# Run failed tests first, then others
pytest --ff
```

### Debugging
```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger on error
pytest --pdbcls=IPython.terminal.debugger:TerminalPdb

# Show full traceback
pytest --tb=long

# Show short traceback
pytest --tb=short

# Show only assertion errors
pytest --tb=line
```

### Performance
```bash
# Show slowest 10 tests
pytest --durations=10

# Show all test durations
pytest --durations=0

# Run in parallel (requires pytest-xdist)
pytest -n auto
pytest -n 4  # Use 4 workers
```

## Pytest Configuration (pytest.ini)

Already created! Located at project root.

Key settings:
- `testpaths = tests .` - Look for tests in tests/ and root
- `python_files = test_*.py` - Test files start with test_
- `markers` - Custom markers for organizing tests
- `addopts` - Default command line options

## Fixtures (conftest.py)

Already created! Fixtures available in all tests:

### Configuration
- `sample_config` - Full test configuration
- `minimal_config` - Minimal configuration

### Data
- `sample_price_data` - Price DataFrame
- `sample_returns` - Returns series
- `sample_price_data_with_metrics` - Price data with metrics

### Models
- `sample_ticker_analysis` - TickerAnalysis object
- `sample_technical_indicators` - Technical indicators
- `sample_advanced_metrics` - Advanced metrics
- `sample_portfolio_metrics` - Portfolio metrics

### Mocks
- `mock_yfinance_ticker` - Mock yfinance (no network)
- `mock_openai_chat` - Mock OpenAI (no API calls)

### Utilities
- `temp_cache_dir` - Temporary cache directory
- `temp_output_dir` - Temporary output directory
- `freeze_time` - Freeze time for tests

### Usage
```python
def test_my_function(sample_config, mock_yfinance_ticker):
    """Fixtures are automatically injected."""
    # Use fixtures directly
    assert sample_config.max_workers == 2
```

## Test Organization

### Current Structure (Transitional)
```
financial_reporting_agent/
├── test_*.py                    # Existing tests (keep for now)
├── tests/                       # New pytest tests
│   ├── unit/                   # Fast, isolated tests
│   ├── integration/            # Component interaction tests
│   └── e2e/                    # Full workflow tests
├── conftest.py                 # Shared fixtures
└── pytest.ini                  # Pytest configuration
```

### Recommended File Naming
- `test_*.py` - Test files
- `test_<component>.py` - Tests for specific component
- `test_integration_*.py` - Integration tests
- `test_e2e_*.py` - End-to-end tests

## Markers Usage

Mark tests to run them selectively:

```python
import pytest

@pytest.mark.unit
def test_fast_function():
    """Fast unit test."""
    pass

@pytest.mark.integration
def test_component_interaction():
    """Integration test."""
    pass

@pytest.mark.slow
def test_performance():
    """Slow performance test."""
    pass

@pytest.mark.network
def test_real_api():
    """Test requiring network."""
    pass

@pytest.mark.llm
def test_openai_integration():
    """Test requiring LLM API."""
    pass
```

Run by marker:
```bash
pytest -m unit              # Only fast tests
pytest -m "not slow"        # Skip slow tests
pytest -m "unit or integration"  # Multiple markers
```

## Tips and Tricks

### 1. Use Fixtures Instead of Setup/Teardown
```python
# Bad (old style)
def setup():
    global cache
    cache = CacheManager()

def teardown():
    cache.clear()

# Good (pytest style)
@pytest.fixture
def cache(temp_cache_dir):
    return CacheManager(cache_dir=temp_cache_dir)
```

### 2. Parametrize for Multiple Test Cases
```python
@pytest.mark.parametrize("ticker,expected", [
    ("aapl", "AAPL"),
    ("msft", "MSFT"),
    ("  GOOGL  ", "GOOGL"),
])
def test_ticker_normalization(ticker, expected):
    assert normalize_ticker(ticker) == expected
```

### 3. Use tmp_path for File Operations
```python
def test_file_creation(tmp_path):
    """tmp_path is automatically cleaned up."""
    file = tmp_path / "test.csv"
    file.write_text("data")
    assert file.exists()
    # No cleanup needed!
```

### 4. Mock External Dependencies
```python
def test_with_mock(monkeypatch):
    """Use monkeypatch to mock."""
    def mock_function():
        return "mocked"

    monkeypatch.setattr("module.function", mock_function)
```

### 5. Test Exceptions
```python
def test_raises_error():
    with pytest.raises(ValueError, match="Invalid ticker"):
        validate_ticker("")
```

## Troubleshooting

### Tests Not Found
```bash
# Check discovery
pytest --collect-only

# Make sure file starts with test_
mv mytest.py test_mytest.py

# Make sure function starts with test_
def test_my_function():  # ✓ Found
def my_test_function():  # ✗ Not found
```

### Import Errors
```bash
# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Fixtures Not Found
```bash
# Make sure conftest.py is in test directory or parent
tests/
├── conftest.py          # ✓ Found by all tests
└── unit/
    └── test_cache.py    # Can use fixtures from conftest.py
```

### Coverage Not Working
```bash
# Install coverage plugin
pip install pytest-cov

# Make sure you're using --cov flag
pytest --cov=.
```

## Next Steps

### 1. Run tests right now (1 minute)
```bash
pytest -v
```

### 2. Check coverage (2 minutes)
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

### 3. Write one test (15 minutes)
Pick one component without tests (cache, alerts, charts) and write 3-5 tests.

### 4. Read the full strategy (30 minutes)
See `TESTING_STRATEGY.md` for comprehensive testing plan.

### 5. Set up CI/CD (30 minutes)
Add GitHub Actions workflow to run tests automatically.

## Resources

- **pytest documentation**: https://docs.pytest.org/
- **pytest fixtures**: https://docs.pytest.org/en/stable/fixture.html
- **pytest markers**: https://docs.pytest.org/en/stable/mark.html
- **pytest-cov**: https://pytest-cov.readthedocs.io/
- **Testing Strategy**: See `TESTING_STRATEGY.md` in this repo

## Summary

You now have:
- ✅ pytest installed and configured
- ✅ Fixtures ready to use (conftest.py)
- ✅ Configuration set up (pytest.ini)
- ✅ Development dependencies (requirements-dev.txt)
- ✅ Quick reference commands

**Run your first pytest command now:**
```bash
pytest -v
```

Good luck! 🚀
