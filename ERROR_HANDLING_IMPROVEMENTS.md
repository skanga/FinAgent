# Error Handling Improvement Suggestions

## Current State Analysis

After analyzing the codebase, here are the current error handling patterns and suggested improvements:

---

## 1. **Custom Exception Hierarchy**

### Current Issues:
- Using generic `ValueError`, `KeyError`, `TypeError` throughout
- No domain-specific exceptions
- Difficult to distinguish between different failure modes
- Catch blocks often group unrelated exceptions together

### Suggested Improvements:

**Create Custom Exception Classes:**
```python
# errors.py (new file)

class FinancialReportError(Exception):
    """Base exception for all financial report errors."""
    pass

class DataFetchError(FinancialReportError):
    """Raised when data fetching fails."""
    def __init__(self, ticker: str, reason: str, original_error: Exception = None):
        self.ticker = ticker
        self.reason = reason
        self.original_error = original_error
        super().__init__(f"Failed to fetch data for {ticker}: {reason}")

class DataValidationError(FinancialReportError):
    """Raised when data validation fails."""
    def __init__(self, ticker: str, validation_type: str, details: str):
        self.ticker = ticker
        self.validation_type = validation_type
        self.details = details
        super().__init__(f"{ticker} validation failed ({validation_type}): {details}")

class AnalysisError(FinancialReportError):
    """Raised when analysis computation fails."""
    def __init__(self, ticker: str, metric: str, reason: str):
        self.ticker = ticker
        self.metric = metric
        self.reason = reason
        super().__init__(f"Analysis failed for {ticker} ({metric}): {reason}")

class LLMError(FinancialReportError):
    """Raised when LLM operations fail."""
    def __init__(self, operation: str, reason: str, retries: int = 0):
        self.operation = operation
        self.reason = reason
        self.retries = retries
        super().__init__(f"LLM {operation} failed after {retries} retries: {reason}")

class CacheError(FinancialReportError):
    """Raised when cache operations fail."""
    pass

class ConfigurationError(FinancialReportError):
    """Raised when configuration is invalid."""
    pass
```

**Benefits:**
- Clear error semantics
- Rich error context (ticker, metric, reason)
- Easy to catch specific error types
- Better error messages
- Structured error data for logging/monitoring

---

## 2. **Retry Mechanisms with Exponential Backoff**

### Current Issues:
- LLM retries exist but are simple (llm_interface.py:139)
- No retries for network errors in fetcher
- No jitter to prevent thundering herd
- Fixed retry count, not configurable

### Suggested Improvements:

**Decorator-Based Retry:**
```python
# retry.py (new file)

from functools import wraps
import time
import random
import logging

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,)
):
    """Retry decorator with exponential backoff and jitter."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        raise

                    # Add jitter to prevent thundering herd
                    actual_delay = delay
                    if jitter:
                        actual_delay *= (0.5 + random.random())

                    logging.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): "
                        f"{e}. Retrying in {actual_delay:.2f}s..."
                    )

                    time.sleep(actual_delay)
                    delay *= backoff_factor

        return wrapper
    return decorator

# Usage in fetcher.py:
@retry_with_backoff(
    max_retries=3,
    initial_delay=1.0,
    exceptions=(OSError, ConnectionError, TimeoutError)
)
def fetch_price_history(self, ticker: str, period: str):
    # ... existing code
```

**Benefits:**
- Handles transient network failures
- Prevents overwhelming failing services
- Configurable per function
- Logging built-in
- Jitter prevents synchronized retries

---

## 3. **Structured Logging with Context**

### Current Issues:
- Basic logging (logger.error, logger.warning)
- No structured context (request_id, ticker, operation)
- Difficult to correlate logs across operations
- No severity levels for different scenarios

### Suggested Improvements:

**Context-Aware Logging:**
```python
# logging_utils.py (new file)

import logging
import contextvars
from typing import Dict, Any

# Thread-local context
log_context = contextvars.ContextVar('log_context', default={})

class ContextAdapter(logging.LoggerAdapter):
    """Logger adapter that adds context to log records."""

    def process(self, msg, kwargs):
        context = log_context.get()
        extra = kwargs.get('extra', {})
        extra.update(context)
        kwargs['extra'] = extra

        # Add context to message
        if context:
            ctx_str = ' '.join(f"{k}={v}" for k, v in context.items())
            msg = f"[{ctx_str}] {msg}"

        return msg, kwargs

def get_logger(name: str) -> ContextAdapter:
    """Get a context-aware logger."""
    return ContextAdapter(logging.getLogger(name), {})

def set_context(**kwargs):
    """Set logging context for current execution."""
    current = log_context.get().copy()
    current.update(kwargs)
    log_context.set(current)

def clear_context():
    """Clear logging context."""
    log_context.set({})

# Usage in orchestrator.py:
logger = get_logger(__name__)

def analyze_ticker(self, ticker: str, period: str):
    set_context(ticker=ticker, period=period, operation='analyze')
    try:
        logger.info("Starting analysis")  # Logs: [ticker=AAPL period=1y operation=analyze] Starting analysis
        # ... analysis code
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise
    finally:
        clear_context()
```

**Benefits:**
- Rich log context
- Easy log correlation
- Better debugging
- Supports distributed tracing
- Clean separation of concerns

---

## 4. **Circuit Breaker Pattern**

### Current Issues:
- No protection against cascading failures
- Continues retrying even when service is clearly down
- Can overwhelm failing APIs
- No automatic recovery detection

### Suggested Improvements:

**Circuit Breaker Implementation:**
```python
# circuit_breaker.py (new file)

from enum import Enum
from datetime import datetime, timedelta
from typing import Callable
import threading

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exceptions: tuple = (Exception,)
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exceptions = expected_exceptions

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker open, will retry after {self.timeout}s"
                    )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exceptions as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout)
        )

    def _on_success(self):
        with self.lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED

    def _on_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

# Usage in fetcher.py:
class CachedDataFetcher:
    def __init__(self, ...):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=60.0,
            expected_exceptions=(DataFetchError, OSError)
        )

    def fetch_price_history(self, ticker: str, period: str):
        return self.circuit_breaker.call(
            self._fetch_price_history_impl,
            ticker,
            period
        )
```

**Benefits:**
- Prevents cascading failures
- Automatic recovery detection
- Fails fast when service is down
- Reduces load on failing services
- Configurable thresholds

---

## 5. **Validation with Explicit Error Messages**

### Current Issues:
- Validation errors buried in generic exceptions
- No clear indication of what failed validation
- Difficult to provide user-friendly error messages

### Suggested Improvements:

**Structured Validation:**
```python
# validation.py (new file)

from dataclasses import dataclass
from typing import List, Optional, Any

@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    field: Optional[str] = None
    value: Optional[Any] = None

    def add_error(self, message: str):
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        self.warnings.append(message)

    @classmethod
    def success(cls):
        return cls(is_valid=True, errors=[], warnings=[])

class Validator:
    """Base validator class."""

    def validate(self, value: Any) -> ValidationResult:
        raise NotImplementedError

class TickerValidator(Validator):
    """Validates ticker symbols."""

    def validate(self, ticker: str) -> ValidationResult:
        result = ValidationResult.success()
        result.field = "ticker"
        result.value = ticker

        if not ticker:
            result.add_error("Ticker cannot be empty")

        if len(ticker) > 10:
            result.add_error(f"Ticker too long: {len(ticker)} chars (max 10)")

        if not ticker.replace('.', '').replace('-', '').isalnum():
            result.add_error(f"Invalid characters in ticker: {ticker}")

        if ticker.lower() in ['test', 'null', 'none']:
            result.add_warning(f"Suspicious ticker name: {ticker}")

        return result

class DataFrameValidator(Validator):
    """Validates DataFrame quality."""

    def __init__(self, min_rows: int = 5):
        self.min_rows = min_rows

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = ValidationResult.success()

        if df.empty:
            result.add_error("DataFrame is empty")
            return result

        if len(df) < self.min_rows:
            result.add_error(
                f"Insufficient data: {len(df)} rows (minimum {self.min_rows})"
            )

        required_columns = ['close', 'Date']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            result.add_error(f"Missing required columns: {missing}")

        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            result.add_warning(f"Null values found: {null_counts.to_dict()}")

        return result

# Usage in fetcher.py:
def fetch_price_history(self, ticker: str, period: str):
    # Validate ticker first
    validation = TickerValidator().validate(ticker)
    if not validation.is_valid:
        raise DataValidationError(
            ticker,
            "ticker_format",
            "; ".join(validation.errors)
        )

    # ... fetch data

    # Validate result
    validation = DataFrameValidator(min_rows=5).validate(hist)
    if not validation.is_valid:
        raise DataValidationError(
            ticker,
            "data_quality",
            "; ".join(validation.errors)
        )

    # Log warnings
    for warning in validation.warnings:
        logger.warning(f"{ticker}: {warning}")
```

**Benefits:**
- Clear validation semantics
- Detailed error messages
- Separation of errors vs warnings
- Reusable validators
- Easy testing

---

## 6. **Graceful Degradation**

### Current Issues:
- All-or-nothing approach in some areas
- Single ticker failure can stop entire report (in some paths)
- No fallback values for missing data

### Suggested Improvements:

**Partial Results Pattern:**
```python
# In analyzers.py:

@dataclass
class AnalysisResult:
    """Result of an analysis with partial success tracking."""
    ticker: str
    data: Optional[TickerAnalysis]
    errors: List[str]
    warnings: List[str]
    partial: bool = False  # True if some metrics failed

    @property
    def success(self) -> bool:
        return self.data is not None

    def add_error(self, error: str):
        self.errors.append(error)

    def add_warning(self, warning: str):
        self.warnings.append(warning)

def analyze_ticker_safe(self, ticker: str) -> AnalysisResult:
    """Analyze ticker with graceful degradation."""
    result = AnalysisResult(ticker=ticker, data=None, errors=[], warnings=[])

    try:
        # Try to get basic data
        price_data = self.fetch_price_history(ticker)

        # Compute metrics with fallbacks
        metrics = AdvancedMetrics()

        try:
            metrics.sharpe_ratio = self._compute_sharpe(price_data)
        except Exception as e:
            result.add_warning(f"Sharpe calculation failed: {e}")
            metrics.sharpe_ratio = None
            result.partial = True

        try:
            metrics.max_drawdown = self._compute_drawdown(price_data)
        except Exception as e:
            result.add_warning(f"Drawdown calculation failed: {e}")
            metrics.max_drawdown = None
            result.partial = True

        # ... other metrics with individual try/except

        result.data = TickerAnalysis(
            ticker=ticker,
            metrics=metrics,
            # ... other fields
        )

    except Exception as e:
        result.add_error(f"Complete analysis failed: {e}")

    return result

# In orchestrator.py:
def run(self, tickers: List[str], period: str):
    results = []
    for ticker in tickers:
        analysis = self.analyzer.analyze_ticker_safe(ticker)
        results.append(analysis)

        if analysis.success:
            logger.info(f"{ticker}: Analysis successful")
        elif analysis.partial:
            logger.warning(f"{ticker}: Partial analysis (some metrics failed)")
        else:
            logger.error(f"{ticker}: Analysis failed completely")

    # Continue with partial results
    successful = [r for r in results if r.success or r.partial]

    if not successful:
        raise ValueError("All analyses failed completely")

    # Generate report with available data
```

**Benefits:**
- Partial failures don't stop entire process
- Rich failure information
- Better user experience
- Transparent about data quality

---

## 7. **Error Aggregation and Reporting**

### Current Issues:
- Errors logged individually
- No summary of all errors
- Difficult to see patterns
- Users don't see full picture of failures

### Suggested Improvements:

**Error Collector:**
```python
# error_collector.py (new file)

from dataclasses import dataclass, field
from typing import List, Dict
from collections import defaultdict

@dataclass
class ErrorSummary:
    """Aggregated error information."""
    total_errors: int = 0
    total_warnings: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_ticker: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    critical_errors: List[str] = field(default_factory=list)

    def add_error(self, error: Exception, ticker: str = None, critical: bool = False):
        self.total_errors += 1
        error_type = type(error).__name__
        self.errors_by_type[error_type] += 1

        if ticker:
            self.errors_by_ticker[ticker].append(str(error))

        if critical:
            self.critical_errors.append(str(error))

    def add_warning(self, warning: str, ticker: str = None):
        self.total_warnings += 1
        if ticker:
            self.errors_by_ticker[ticker].append(f"WARNING: {warning}")

    def has_critical_errors(self) -> bool:
        return len(self.critical_errors) > 0

    def format_summary(self) -> str:
        """Format a human-readable summary."""
        lines = []
        lines.append(f"Errors: {self.total_errors}, Warnings: {self.total_warnings}")

        if self.errors_by_type:
            lines.append("\nErrors by type:")
            for error_type, count in sorted(self.errors_by_type.items()):
                lines.append(f"  - {error_type}: {count}")

        if self.errors_by_ticker:
            lines.append("\nErrors by ticker:")
            for ticker, errors in sorted(self.errors_by_ticker.items()):
                lines.append(f"  - {ticker}: {len(errors)} issue(s)")
                for error in errors[:3]:  # Show first 3
                    lines.append(f"      • {error}")
                if len(errors) > 3:
                    lines.append(f"      ... and {len(errors) - 3} more")

        if self.critical_errors:
            lines.append("\nCRITICAL ERRORS:")
            for error in self.critical_errors:
                lines.append(f"  ❌ {error}")

        return "\n".join(lines)

# Usage in orchestrator.py:
def run(self, tickers: List[str], period: str):
    error_summary = ErrorSummary()

    for ticker in tickers:
        try:
            analysis = self.analyze_ticker(ticker, period)
        except DataFetchError as e:
            error_summary.add_error(e, ticker=ticker)
        except AnalysisError as e:
            error_summary.add_error(e, ticker=ticker)
        except Exception as e:
            error_summary.add_error(e, ticker=ticker, critical=True)

    # Add to report metadata
    print("\n" + error_summary.format_summary())

    if error_summary.has_critical_errors():
        raise RuntimeError("Critical errors occurred during analysis")
```

**Benefits:**
- Comprehensive error overview
- Pattern detection (e.g., all Yahoo API errors)
- Better user communication
- Helps identify systemic issues

---

## 8. **Timeout Management**

### Current Issues:
- Fixed timeouts in fetcher (30s)
- No granular timeout control
- Slow operations can block entire process
- No timeout for LLM calls beyond config

### Suggested Improvements:

**Configurable Timeouts:**
```python
# timeouts.py (new file)

from dataclasses import dataclass
from typing import Optional
import signal
from contextlib import contextmanager

@dataclass
class TimeoutConfig:
    """Centralized timeout configuration."""
    http_connect: float = 10.0
    http_read: float = 30.0
    llm_call: float = 60.0
    analysis: float = 120.0
    total_report: float = 600.0

class TimeoutError(FinancialReportError):
    """Raised when operation exceeds timeout."""
    pass

@contextmanager
def timeout(seconds: float, operation: str = "operation"):
    """Context manager for operation timeout."""

    def timeout_handler(signum, frame):
        raise TimeoutError(f"{operation} exceeded {seconds}s timeout")

    # Set alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# Usage:
def analyze_ticker(self, ticker: str):
    with timeout(self.config.timeouts.analysis, f"Analysis for {ticker}"):
        # ... analysis code
```

**Benefits:**
- Prevents hanging operations
- Configurable per operation type
- Clear timeout errors
- Better resource management

---

## 9. **Health Checks and Self-Diagnostics**

### Current Issues:
- No way to check system health before running
- API keys validated only on first call
- Network issues discovered mid-execution

### Suggested Improvements:

**Health Check System:**
```python
# health.py (new file)

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str
    latency_ms: Optional[float] = None

class HealthChecker:
    """System health checker."""

    def check_all(self) -> List[HealthCheck]:
        """Run all health checks."""
        checks = []

        checks.append(self._check_config())
        checks.append(self._check_yfinance())
        checks.append(self._check_llm())
        checks.append(self._check_cache())
        checks.append(self._check_disk_space())

        return checks

    def _check_config(self) -> HealthCheck:
        """Check configuration."""
        try:
            config = Config.from_env()
            if not config.openai_api_key:
                return HealthCheck(
                    "config",
                    HealthStatus.UNHEALTHY,
                    "OpenAI API key not configured"
                )
            return HealthCheck("config", HealthStatus.HEALTHY, "Configuration valid")
        except Exception as e:
            return HealthCheck("config", HealthStatus.UNHEALTHY, str(e))

    def _check_yfinance(self) -> HealthCheck:
        """Check yfinance connectivity."""
        import time
        start = time.time()
        try:
            ticker = yf.Ticker("AAPL")
            ticker.info  # Quick API call
            latency = (time.time() - start) * 1000
            return HealthCheck(
                "yfinance",
                HealthStatus.HEALTHY,
                f"Yahoo Finance API accessible",
                latency_ms=latency
            )
        except Exception as e:
            return HealthCheck("yfinance", HealthStatus.UNHEALTHY, str(e))

    def _check_llm(self) -> HealthCheck:
        """Check LLM API."""
        try:
            config = Config.from_env()
            llm = IntegratedLLMInterface(config)
            # Simple test call
            response = llm.llm.invoke("Test")
            return HealthCheck("llm", HealthStatus.HEALTHY, "LLM API accessible")
        except Exception as e:
            return HealthCheck("llm", HealthStatus.UNHEALTHY, str(e))

    def _check_cache(self) -> HealthCheck:
        """Check cache directory."""
        try:
            cache_dir = Path(".cache")
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True)

            # Test write
            test_file = cache_dir / ".health_check"
            test_file.write_text("test")
            test_file.unlink()

            return HealthCheck("cache", HealthStatus.HEALTHY, "Cache directory writable")
        except Exception as e:
            return HealthCheck("cache", HealthStatus.UNHEALTHY, str(e))

    def _check_disk_space(self) -> HealthCheck:
        """Check available disk space."""
        import shutil
        try:
            usage = shutil.disk_usage(".")
            free_gb = usage.free / (1024**3)

            if free_gb < 0.1:
                status = HealthStatus.UNHEALTHY
                message = f"Critically low disk space: {free_gb:.2f}GB"
            elif free_gb < 1.0:
                status = HealthStatus.DEGRADED
                message = f"Low disk space: {free_gb:.2f}GB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Sufficient disk space: {free_gb:.2f}GB"

            return HealthCheck("disk", status, message)
        except Exception as e:
            return HealthCheck("disk", HealthStatus.UNHEALTHY, str(e))

# Usage in main.py:
def main():
    # Run health checks first
    print("Running health checks...")
    checker = HealthChecker()
    checks = checker.check_all()

    for check in checks:
        status_icon = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}
        print(f"{status_icon[check.status.value]} {check.component}: {check.message}")

    unhealthy = [c for c in checks if c.status == HealthStatus.UNHEALTHY]
    if unhealthy:
        print(f"\n❌ System unhealthy. Fix issues before continuing.")
        return 1

    # Continue with report generation
```

**Benefits:**
- Early problem detection
- Clear system status
- Better user guidance
- Reduced wasted execution time

---

## 10. **Error Recovery Strategies**

### Current Issues:
- Limited recovery options
- No automatic fallbacks
- User must manually retry

### Suggested Improvements:

**Recovery Manager:**
```python
# recovery.py (new file)

from typing import Callable, Any, List
from dataclasses import dataclass

@dataclass
class RecoveryStrategy:
    """A recovery strategy for a failed operation."""
    name: str
    action: Callable[[], Any]
    description: str

class RecoveryManager:
    """Manages error recovery strategies."""

    def try_recover(self, error: Exception, strategies: List[RecoveryStrategy]) -> Any:
        """Try recovery strategies in order."""

        logger.warning(f"Attempting recovery from: {error}")

        for strategy in strategies:
            try:
                logger.info(f"Trying recovery strategy: {strategy.name}")
                result = strategy.action()
                logger.info(f"Recovery successful: {strategy.name}")
                return result
            except Exception as e:
                logger.warning(f"Recovery strategy '{strategy.name}' failed: {e}")

        # All strategies failed
        raise error

# Usage in fetcher.py:
def fetch_price_history(self, ticker: str, period: str):
    recovery = RecoveryManager()

    try:
        return self._fetch_from_yfinance(ticker, period)
    except DataFetchError as e:
        # Try recovery strategies
        strategies = [
            RecoveryStrategy(
                "cache_fallback",
                lambda: self._fetch_from_cache_any_period(ticker),
                "Use cached data from any period"
            ),
            RecoveryStrategy(
                "shorter_period",
                lambda: self._fetch_from_yfinance(ticker, "1mo"),
                "Fetch shorter time period (1 month)"
            ),
            RecoveryStrategy(
                "minimal_data",
                lambda: self._fetch_minimal_data(ticker),
                "Fetch only current price data"
            ),
        ]

        return recovery.try_recover(e, strategies)
```

**Benefits:**
- Automatic recovery attempts
- Transparent fallback chain
- Better success rate
- Logged recovery paths

---

## Priority Implementation Order

1. **High Priority** (Immediate impact):
   - Custom exception hierarchy
   - Structured validation
   - Error aggregation and reporting
   - Health checks

2. **Medium Priority** (Significant improvement):
   - Retry with exponential backoff
   - Circuit breaker
   - Graceful degradation
   - Structured logging

3. **Lower Priority** (Nice to have):
   - Timeout management
   - Error recovery strategies

---

## Summary

These improvements would transform error handling from **reactive** to **proactive**:

- **Better diagnostics** through structured exceptions and logging
- **Improved resilience** through retries, circuit breakers, and graceful degradation
- **Enhanced visibility** through error aggregation and health checks
- **Better user experience** through clear error messages and recovery strategies

All suggestions are designed to be **incrementally adoptable** - you can implement them one at a time without breaking existing functionality.
