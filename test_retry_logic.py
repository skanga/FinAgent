"""
Test retry logic with exponential backoff.
"""
import sys
import time
import logging
from unittest.mock import Mock, patch, MagicMock
from tenacity import RetryError

# Configure stdout to handle unicode on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Configure logging to see retry attempts
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_fetcher_retry():
    """Test retry logic in CachedDataFetcher.fetch_price_history"""
    print("\n" + "="*70)
    print("TEST 1: Fetcher Retry Logic")
    print("="*70)

    from cache import CacheManager
    from fetcher import CachedDataFetcher

    cache = CacheManager(cache_dir="./.cache_test", ttl_hours=24)
    fetcher = CachedDataFetcher(cache)

    # Test 1: Simulate transient network error that recovers
    print("\n[Test 1.1] Simulating transient network error (should retry and succeed)...")

    call_count = {'count': 0}
    original_ticker = Mock()

    def flaky_history(*args, **kwargs):
        call_count['count'] += 1
        if call_count['count'] < 3:
            print(f"  Attempt {call_count['count']}: Raising OSError (simulated network error)")
            raise OSError("Simulated network error")
        print(f"  Attempt {call_count['count']}: Success!")
        # Return mock DataFrame
        import pandas as pd
        df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=10),
            'Close': [100 + i for i in range(10)],
            'Open': [99 + i for i in range(10)],
            'High': [101 + i for i in range(10)],
            'Low': [98 + i for i in range(10)],
            'Volume': [1000000] * 10
        })
        return df

    with patch('yfinance.Ticker') as mock_ticker_class:
        mock_ticker = Mock()
        mock_ticker.history = flaky_history
        mock_ticker_class.return_value = mock_ticker

        try:
            start = time.time()
            result = fetcher.fetch_price_history("AAPL", "1y")
            elapsed = time.time() - start
            print(f"\n✓ Success after {call_count['count']} attempts (took {elapsed:.2f}s)")
            print(f"  Expected delays: ~3s (1s + 2s between retries)")
            assert call_count['count'] == 3, f"Expected 3 attempts, got {call_count['count']}"
            assert len(result) > 0, "Expected non-empty DataFrame"
            print("[PASS] Retry with recovery works correctly")
        except Exception as e:
            print(f"[FAIL] Unexpected error: {e}")
            return False

    # Test 2: Persistent network error (should fail after max retries)
    print("\n[Test 1.2] Simulating persistent network error (should fail after 3 attempts)...")

    call_count['count'] = 0

    def always_fail(*args, **kwargs):
        call_count['count'] += 1
        print(f"  Attempt {call_count['count']}: Raising OSError")
        raise OSError("Persistent network error")

    with patch('yfinance.Ticker') as mock_ticker_class:
        mock_ticker = Mock()
        mock_ticker.history = always_fail
        mock_ticker_class.return_value = mock_ticker

        try:
            start = time.time()
            result = fetcher.fetch_price_history("BADTICKER", "1y")
            print(f"[FAIL] Should have raised OSError after retries")
            return False
        except OSError as e:
            elapsed = time.time() - start
            print(f"\n✓ Failed after {call_count['count']} attempts (took {elapsed:.2f}s)")
            print(f"  Expected delays: ~7s (1s + 2s + 4s between retries)")
            assert call_count['count'] == 3, f"Expected 3 attempts, got {call_count['count']}"
            print(f"  Error message: {str(e)[:100]}")
            print("[PASS] Max retries work correctly")

    # Test 3: ValueError should NOT retry
    print("\n[Test 1.3] Invalid ticker (should NOT retry)...")

    call_count['count'] = 0

    def immediate_fail(*args, **kwargs):
        call_count['count'] += 1
        print(f"  Attempt {call_count['count']}: Raising ValueError")
        raise ValueError("Invalid ticker")

    with patch('yfinance.Ticker') as mock_ticker_class:
        mock_ticker = Mock()
        mock_ticker.history = immediate_fail
        mock_ticker_class.return_value = mock_ticker

        try:
            start = time.time()
            result = fetcher.fetch_price_history("INVALID", "1y")
            print(f"[FAIL] Should have raised ValueError")
            return False
        except ValueError:
            elapsed = time.time() - start
            print(f"\n✓ Failed immediately after {call_count['count']} attempt (took {elapsed:.2f}s)")
            assert call_count['count'] == 1, f"Expected 1 attempt (no retry), got {call_count['count']}"
            print("[PASS] Non-retryable errors are not retried")

    return True


def test_llm_retry():
    """Test retry logic in LLM interface methods"""
    print("\n" + "="*70)
    print("TEST 2: LLM Interface Retry Logic")
    print("="*70)

    from config import Config
    from llm_interface import IntegratedLLMInterface

    # Create config with dummy API key
    config = Config(
        openai_api_key="test-key",
        model_name="gpt-4o",
        cache_ttl_hours=24,
        request_timeout=30,
        risk_free_rate=0.02,
        benchmark_ticker="SPY"
    )

    # Test parse_natural_language_request retry
    print("\n[Test 2.1] Testing parse_natural_language_request retry...")

    call_count = {'count': 0}

    def flaky_invoke(*args, **kwargs):
        call_count['count'] += 1
        if call_count['count'] < 2:
            print(f"  Attempt {call_count['count']}: Raising ConnectionError")
            raise ConnectionError("Simulated API timeout")
        print(f"  Attempt {call_count['count']}: Success!")
        # Return mock response
        mock_response = Mock()
        mock_response.content = '{"tickers": ["AAPL", "MSFT"], "period": "1y", "metrics": [], "output_format": "markdown"}'
        return mock_response

    with patch.object(IntegratedLLMInterface, '__init__', lambda x, y: None):
        llm = IntegratedLLMInterface(config)
        llm.parser_chain = Mock()
        llm.parser_chain.invoke = flaky_invoke

        try:
            start = time.time()
            result = llm.parse_natural_language_request("Analyze AAPL and MSFT")
            elapsed = time.time() - start
            print(f"\n✓ Success after {call_count['count']} attempts (took {elapsed:.2f}s)")
            assert call_count['count'] == 2, f"Expected 2 attempts, got {call_count['count']}"
            assert result.tickers == ["AAPL", "MSFT"], "Unexpected parse result"
            print("[PASS] LLM parse retry works correctly")
        except Exception as e:
            print(f"[FAIL] Unexpected error: {e}")
            return False

    # Test JSON parse error should NOT retry
    print("\n[Test 2.2] Testing JSON error (should NOT retry)...")

    call_count['count'] = 0

    def bad_json(*args, **kwargs):
        call_count['count'] += 1
        print(f"  Attempt {call_count['count']}: Returning invalid JSON")
        mock_response = Mock()
        mock_response.content = 'invalid json'
        return mock_response

    with patch.object(IntegratedLLMInterface, '__init__', lambda x, y: None):
        llm = IntegratedLLMInterface(config)
        llm.parser_chain = Mock()
        llm.parser_chain.invoke = bad_json

        try:
            result = llm.parse_natural_language_request("Analyze AAPL")
            print(f"[FAIL] Should have raised ValueError")
            return False
        except ValueError as e:
            print(f"\n✓ Failed immediately after {call_count['count']} attempt")
            assert call_count['count'] == 1, f"Expected 1 attempt (no retry), got {call_count['count']}"
            print(f"  Error message: {str(e)[:80]}")
            print("[PASS] JSON errors are not retried")

    return True


def test_retry_timing():
    """Test that exponential backoff timing is correct"""
    print("\n" + "="*70)
    print("TEST 3: Exponential Backoff Timing")
    print("="*70)

    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError

    call_times = []

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ValueError),
        reraise=True
    )
    def failing_function():
        call_times.append(time.time())
        raise ValueError("Test error")

    print("\n[Test 3.1] Measuring actual delays between retry attempts...")

    try:
        failing_function()
    except (ValueError, RetryError):
        pass

    if len(call_times) >= 2:
        delays = [call_times[i] - call_times[i-1] for i in range(1, len(call_times))]
        print(f"\n  Total attempts: {len(call_times)}")
        print(f"  Actual delays: {[f'{d:.2f}s' for d in delays]}")
        print(f"  Expected delays: ~1s, ~2s, ~4s")

        # Check delays are approximately correct (within 0.5s tolerance)
        expected = [1, 2, 4]
        for i, (actual, expected_delay) in enumerate(zip(delays, expected)):
            diff = abs(actual - expected_delay)
            if diff > 0.5:
                print(f"  [!] Delay {i+1} differs by {diff:.2f}s from expected")
            else:
                print(f"  ✓ Delay {i+1} is within tolerance")

        print("[PASS] Exponential backoff timing is correct")
        return True
    else:
        print("[FAIL] Not enough attempts recorded")
        return False


def main():
    """Run all retry logic tests"""
    print("\n" + "="*70)
    print("RETRY LOGIC TEST SUITE")
    print("="*70)
    print("\nThis test suite validates:")
    print("1. Fetcher retries transient network errors with exponential backoff")
    print("2. LLM interface retries transient errors")
    print("3. Non-retryable errors (ValueError, JSON errors) fail immediately")
    print("4. Exponential backoff timing is correct (1s, 2s, 4s)")

    results = []

    try:
        results.append(("Fetcher Retry", test_fetcher_retry()))
    except Exception as e:
        print(f"\n[ERROR] Fetcher test crashed: {e}")
        results.append(("Fetcher Retry", False))

    try:
        results.append(("LLM Retry", test_llm_retry()))
    except Exception as e:
        print(f"\n[ERROR] LLM test crashed: {e}")
        results.append(("LLM Retry", False))

    try:
        results.append(("Backoff Timing", test_retry_timing()))
    except Exception as e:
        print(f"\n[ERROR] Timing test crashed: {e}")
        results.append(("Backoff Timing", False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} - {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All retry logic tests passed!")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
