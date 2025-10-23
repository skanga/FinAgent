# Retry Behavior Documentation

## Overview

The Financial Reporting Agent uses the **tenacity** library to provide robust retry logic for network operations. This ensures resilience against temporary failures when fetching financial data from external APIs.

## Why Retry Logic?

Financial data APIs can experience:
- **Temporary network failures** (connection drops, timeouts)
- **Rate limiting** (429 Too Many Requests)
- **Server errors** (500, 502, 503, 504)
- **Transient DNS issues**

Without retry logic, a single network hiccup would cause the entire analysis to fail. With retry logic, the system automatically recovers from temporary failures.

## Retry Configuration

### Default Retry Strategy

The system uses **exponential backoff** with the following parameters:

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

@retry(
    stop=stop_after_attempt(3),              # Maximum 3 attempts
    wait=wait_exponential(                   # Exponential backoff
        multiplier=1,                        # Base multiplier
        min=2,                               # Minimum wait: 2 seconds
        max=10                               # Maximum wait: 10 seconds
    ),
    retry_if_exception_type=(                # Retry these exceptions
        OSError,                             # Network errors
        ConnectionError,                     # Connection failures
        TimeoutError,                        # Timeout errors
    ),
    reraise=True                             # Re-raise if all attempts fail
)
def fetch_with_retry():
    # Network operation here
    ...
```

### Backoff Calculation

The wait time between retries follows this pattern:

```
Attempt 1: Initial request (no wait)
↓ fails
Wait: min(2^1 * 1, 10) = 2 seconds

Attempt 2: Retry after 2 seconds
↓ fails
Wait: min(2^2 * 1, 10) = 4 seconds

Attempt 3: Final retry after 4 seconds
↓ fails
Error raised to caller
```

**Total worst-case time:** Initial + 2s + retry + 4s + retry = ~6 seconds + request times

## Implementation Locations

### 1. Data Fetcher (fetcher.py)

#### `fetch_price_history()`

**Retry Configuration:**
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_if_exception_type=(OSError, ConnectionError, TimeoutError),
    reraise=True
)
def fetch_price_history(
    self,
    ticker: str,
    period: str = "1y"
) -> pd.DataFrame:
    """
    Fetches historical price data with automatic retry on network errors.

    Retries up to 3 times with exponential backoff (2s, 4s).
    """
```

**What Gets Retried:**
- yfinance API calls (`yf.Ticker().history()`)
- Network timeouts
- Connection failures
- DNS resolution errors

**What Does NOT Get Retried:**
- Invalid ticker symbols (ValueError) - permanent failure
- Data parsing errors (KeyError) - permanent failure
- Invalid period (ValueError) - permanent failure

**Example Usage:**
```python
# This will automatically retry on network errors
fetcher = CachedDataFetcher(cache_ttl_hours=24)

try:
    data = fetcher.fetch_price_history("AAPL", "1y")
    # Success on first try or after retries
except OSError as e:
    # All 3 attempts failed
    logger.error(f"Failed to fetch data after retries: {e}")
```

### 2. LLM Interface (llm_interface.py)

#### `parse_natural_language_request()`

**Retry Configuration:**
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_if_exception_type=(OSError, ConnectionError, TimeoutError),
    reraise=True
)
def parse_natural_language_request(
    self,
    user_request: str
) -> ParsedRequest:
    """
    Parses natural language with automatic retry on network errors.

    Retries LLM API calls up to 3 times with exponential backoff.
    """
```

**What Gets Retried:**
- LLM API connection errors
- Network timeouts
- Rate limit errors (if wrapped in OSError)

**What Does NOT Get Retried:**
- Invalid JSON responses - uses manual retry loop (max 3)
- Validation errors - permanent failure
- Empty requests - permanent failure

**Example Usage:**
```python
llm = IntegratedLLMInterface(config)

try:
    parsed = llm.parse_natural_language_request(
        "Compare AAPL and MSFT over the past year"
    )
    # Returns ParsedRequest with tickers=["AAPL", "MSFT"], period="1y"
except ConnectionError:
    # All retries exhausted
    print("Unable to reach LLM service")
```

## Error Handling Integration

The retry logic integrates with the orchestrator's error handling using match/case:

```python
# In orchestrator.py
try:
    raw_price_history = self.fetcher.fetch_price_history(ticker, period)
except Exception as e:
    match e:
        case OSError() | ConnectionError() | TimeoutError():
            # This means ALL retry attempts failed
            error_msg = f"Network error for {ticker}: {e}"
            logger.warning(f"NETWORK ERROR (retryable) - {error_msg}")
            # Returns error analysis, but marks as potentially retryable

        case ValueError():
            # Validation error - no retry attempted
            error_msg = f"Invalid ticker {ticker}: {e}"
            logger.error(f"VALIDATION ERROR - {error_msg}")
```

**Key Insight:** If a network exception reaches the orchestrator's error handler, it means the retry logic already attempted 3 times and failed.

## Retry Logging

### Tenacity Logging

Enable detailed retry logging:

```python
import logging

# Enable tenacity's retry logging
logging.getLogger("tenacity").setLevel(logging.DEBUG)

# You'll see logs like:
# DEBUG:tenacity.retry:Starting call to 'fetch_price_history', this is the 1st time calling it.
# DEBUG:tenacity.retry:Retrying 'fetch_price_history' in 2.0 seconds as it raised ConnectionError.
# DEBUG:tenacity.retry:Starting call to 'fetch_price_history', this is the 2nd time calling it.
```

### Application Logging

The application logs retry-related information:

```python
# Before retry (in fetcher.py)
logger.debug(f"Fetching price history for {ticker}")

# After successful retry
logger.info(f"Successfully fetched {ticker} data")

# After all retries fail (in orchestrator.py)
logger.warning(f"NETWORK ERROR (retryable) - Network error for {ticker}: {e}")
```

## Performance Considerations

### Cache First, Retry Second

The retry logic is **inside** the cache check:

```python
def fetch_price_history(self, ticker: str, period: str):
    cache_key = self.cache.get_cache_key(ticker, period, "price_history")

    # Check cache FIRST (no network call)
    cached_data = self.cache.get(cache_key)
    if cached_data is not None:
        return cached_data  # No retry needed

    # Cache miss - now do network call WITH retry
    @retry(...)
    def _fetch():
        return yf.Ticker(ticker).history(period=period)

    data = _fetch()
    self.cache.set(cache_key, data)
    return data
```

**Benefits:**
1. Cache hits skip retry logic entirely
2. Only fresh API calls experience retry overhead
3. Maximum retry overhead: ~6 seconds for 3 attempts
4. Typical case: 0 seconds (cache hit)

### Timeout Configuration

Each retry attempt has its own timeout:

```python
# In config.py
REQUEST_TIMEOUT = 30  # seconds per attempt

# Worst case with retries:
# Attempt 1: 30s timeout
# Wait: 2s
# Attempt 2: 30s timeout
# Wait: 4s
# Attempt 3: 30s timeout
# Total: up to 96 seconds
```

To reduce worst-case time, adjust the timeout:

```python
# In .env
REQUEST_TIMEOUT=15  # Reduces worst case to 51 seconds
```

## Customizing Retry Behavior

### Option 1: Environment Variables

Adjust the number of workers to handle more concurrent retries:

```bash
# .env file
MAX_WORKERS=5  # More workers = more concurrent API calls
REQUEST_TIMEOUT=20  # Shorter timeout = faster failure detection
```

### Option 2: Modify Retry Decorators

Edit `fetcher.py` to change retry behavior:

```python
# More aggressive retries
@retry(
    stop=stop_after_attempt(5),  # 5 attempts instead of 3
    wait=wait_exponential(multiplier=1, min=1, max=5),  # Faster backoff
    retry_if_exception_type=(OSError, ConnectionError, TimeoutError),
)

# Or more conservative (give up faster)
@retry(
    stop=stop_after_attempt(2),  # Only 2 attempts
    wait=wait_fixed(3),  # Fixed 3 second wait
    retry_if_exception_type=(ConnectionError,),  # Only retry connection errors
)
```

### Option 3: Add Retry Callbacks

Monitor retry attempts:

```python
from tenacity import retry, before_sleep_log, after_log

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_if_exception_type=(OSError, ConnectionError, TimeoutError),
    before_sleep=before_sleep_log(logger, logging.WARNING),  # Log before retry
    after=after_log(logger, logging.INFO),  # Log after attempt
)
def fetch_price_history(...):
    ...
```

## Testing Retry Behavior

### Unit Testing with Mocks

```python
import pytest
from unittest.mock import Mock, patch
from fetcher import CachedDataFetcher

def test_fetch_retries_on_connection_error():
    """Test that fetch retries on ConnectionError."""
    fetcher = CachedDataFetcher()

    with patch("yfinance.Ticker") as mock_ticker:
        # Fail twice, succeed on third attempt
        mock_ticker.return_value.history.side_effect = [
            ConnectionError("Network down"),
            ConnectionError("Still down"),
            pd.DataFrame({"Close": [100, 101, 102]}),  # Success
        ]

        # Should succeed after 2 retries
        result = fetcher.fetch_price_history("AAPL", "1y")

        # Verify it was called 3 times
        assert mock_ticker.return_value.history.call_count == 3
        assert len(result) == 3


def test_fetch_gives_up_after_max_retries():
    """Test that fetch gives up after 3 attempts."""
    fetcher = CachedDataFetcher()

    with patch("yfinance.Ticker") as mock_ticker:
        # Always fail
        mock_ticker.return_value.history.side_effect = ConnectionError("Permanent failure")

        # Should raise after 3 attempts
        with pytest.raises(ConnectionError):
            fetcher.fetch_price_history("AAPL", "1y")

        # Verify it tried 3 times
        assert mock_ticker.return_value.history.call_count == 3
```

### Integration Testing

Test with actual network conditions:

```python
def test_fetch_with_slow_network(tmp_path):
    """Test retry behavior with actual slow responses."""
    import time

    fetcher = CachedDataFetcher(cache_dir=tmp_path)

    # This may retry if network is slow
    start = time.time()
    try:
        data = fetcher.fetch_price_history("AAPL", "1d")
        elapsed = time.time() - start

        if elapsed > 5:
            print(f"Network was slow, took {elapsed:.1f}s (possibly retried)")
        else:
            print(f"Network was fast, took {elapsed:.1f}s (no retry needed)")

    except (OSError, ConnectionError) as e:
        elapsed = time.time() - start
        print(f"All retries failed after {elapsed:.1f}s: {e}")
```

## Best Practices

### 1. Don't Retry Too Aggressively
- **Bad:** 10 attempts with no backoff → hammers failing service
- **Good:** 3 attempts with exponential backoff → gives service time to recover

### 2. Retry Only Transient Errors
- **Retry:** Network errors, timeouts, 503 Service Unavailable
- **Don't Retry:** Validation errors, 404 Not Found, 401 Unauthorized

### 3. Log Retry Attempts
- Always log when retrying
- Include attempt number and wait time
- Help debug intermittent failures

### 4. Set Reasonable Timeouts
- Too short → unnecessary retries
- Too long → slow failure detection
- Sweet spot: 15-30 seconds for API calls

### 5. Use Circuit Breaker Pattern (Future Enhancement)
- If service is consistently failing, stop retrying
- Wait longer before trying again
- Prevents cascade failures

## Troubleshooting

### "Operation timed out after 3 retries"

**Cause:** All retry attempts exceeded the timeout.

**Solutions:**
1. Check network connectivity: `ping api.openai.com`
2. Increase timeout: `REQUEST_TIMEOUT=45` in .env
3. Check API service status
4. Try fewer concurrent workers: `MAX_WORKERS=2`

### "Rate limited by API"

**Cause:** Too many requests in short time.

**Solutions:**
1. Enable caching: Reduce duplicate requests
2. Reduce concurrent workers: `MAX_WORKERS=2`
3. Add delay between requests
4. Check API rate limits

### "Retries seem to be ignored"

**Cause:** Error is not in the retry exception types.

**Solutions:**
1. Check exception type: Only OSError, ConnectionError, TimeoutError are retried
2. Wrap custom exceptions if needed
3. Add exception type to retry decorator

## Related Documentation

- [Architecture](ARCHITECTURE.md) - System overview
- [Configuration](CONFIG_EXAMPLES.md) - Custom config usage
- [CLAUDE.md](CLAUDE.md) - Development guide
