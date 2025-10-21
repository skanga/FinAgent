"""
Test input validation improvements in main.py
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import validate_tickers, validate_period, parse_weights


def test_weight_validation():
    """Test weight validation catches negative numbers and other issues."""
    print("Testing weight validation...")

    # Test 1: Negative weights (SHOULD FAIL)
    print("\n[TEST] Negative weight should fail")
    try:
        parse_weights("-0.5,0.8,0.7", ["AAPL", "MSFT", "GOOGL"])
        print("  [FAIL] Should have raised ValueError for negative weight")
        return False
    except ValueError as e:
        if "negative" in str(e).lower():
            print(f"  [OK] Correctly rejected negative weight: {e}")
        else:
            print(f"  [FAIL] Wrong error message: {e}")
            return False

    # Test 2: Weight > 1.0 (SHOULD FAIL)
    print("\n[TEST] Weight > 1.0 should fail")
    try:
        parse_weights("0.5,1.5,0.2", ["AAPL", "MSFT", "GOOGL"])
        print("  [FAIL] Should have raised ValueError for weight > 1.0")
        return False
    except ValueError as e:
        if ">1.0" in str(e) or "greater than" in str(e).lower():
            print(f"  [OK] Correctly rejected weight > 1.0: {e}")
        else:
            print(f"  [FAIL] Wrong error message: {e}")
            return False

    # Test 3: Weights don't sum to 1.0 (SHOULD FAIL)
    print("\n[TEST] Weights not summing to 1.0 should fail")
    try:
        parse_weights("0.3,0.3,0.3", ["AAPL", "MSFT", "GOOGL"])
        print("  [FAIL] Should have raised ValueError for weights not summing to 1.0")
        return False
    except ValueError as e:
        if "sum" in str(e).lower():
            print(f"  [OK] Correctly rejected weights not summing to 1.0: {e}")
        else:
            print(f"  [FAIL] Wrong error message: {e}")
            return False

    # Test 4: Wrong number of weights (SHOULD FAIL)
    print("\n[TEST] Wrong number of weights should fail")
    try:
        parse_weights("0.5,0.5", ["AAPL", "MSFT", "GOOGL"])
        print("  [FAIL] Should have raised ValueError for wrong number of weights")
        return False
    except ValueError as e:
        if "number of weights" in str(e).lower() or "must match" in str(e).lower():
            print(f"  [OK] Correctly rejected wrong count: {e}")
        else:
            print(f"  [FAIL] Wrong error message: {e}")
            return False

    # Test 5: Valid weights (SHOULD PASS)
    print("\n[TEST] Valid weights should pass")
    try:
        result = parse_weights("0.5,0.3,0.2", ["AAPL", "MSFT", "GOOGL"])
        if result == {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}:
            print(f"  [OK] Valid weights accepted: {result}")
        else:
            print(f"  [FAIL] Unexpected result: {result}")
            return False
    except ValueError as e:
        print(f"  [FAIL] Valid weights rejected: {e}")
        return False

    # Test 6: Weights with tolerance (SHOULD PASS)
    print("\n[TEST] Weights within tolerance should pass")
    try:
        result = parse_weights("0.333,0.333,0.334", ["AAPL", "MSFT", "GOOGL"])
        print(f"  [OK] Weights within tolerance accepted: {result}")
    except ValueError as e:
        print(f"  [FAIL] Weights within tolerance rejected: {e}")
        return False

    # Test 7: Multiple negative weights (SHOULD FAIL with good error message)
    print("\n[TEST] Multiple invalid weights should show all issues")
    try:
        parse_weights("-0.5,1.5,0.3", ["AAPL", "MSFT", "GOOGL"])
        print("  [FAIL] Should have raised ValueError for multiple invalid weights")
        return False
    except ValueError as e:
        error_str = str(e)
        if "negative" in error_str.lower() and ("1.5" in error_str or ">1.0" in error_str):
            print(f"  [OK] Shows multiple issues: {e}")
        elif "negative" in error_str.lower():
            print(f"  [OK] At least caught negative weight: {e}")
        else:
            print(f"  [FAIL] Wrong error message: {e}")
            return False

    print("\n[OK] All weight validation tests passed")
    return True


def test_period_validation():
    """Test period validation in code (not just argparse)."""
    print("\n\nTesting period validation...")

    # Test 1: Valid periods (SHOULD PASS)
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    print(f"\n[TEST] Valid periods should pass")
    for period in valid_periods:
        try:
            validate_period(period)
        except ValueError as e:
            print(f"  [FAIL] Valid period '{period}' rejected: {e}")
            return False
    print(f"  [OK] All {len(valid_periods)} valid periods accepted")

    # Test 2: Invalid period (SHOULD FAIL)
    print("\n[TEST] Invalid period should fail")
    try:
        validate_period("1w")  # Not a valid yfinance period
        print("  [FAIL] Should have raised ValueError for invalid period '1w'")
        return False
    except ValueError as e:
        if "invalid period" in str(e).lower() or "must be one of" in str(e).lower():
            print(f"  [OK] Correctly rejected invalid period: {e}")
        else:
            print(f"  [FAIL] Wrong error message: {e}")
            return False

    # Test 3: Invalid period with typo (SHOULD FAIL)
    print("\n[TEST] Period with typo should fail")
    try:
        validate_period("1yr")  # Common typo
        print("  [FAIL] Should have raised ValueError for invalid period '1yr'")
        return False
    except ValueError as e:
        if "invalid period" in str(e).lower() or "must be one of" in str(e).lower():
            print(f"  [OK] Correctly rejected typo: {e}")
        else:
            print(f"  [FAIL] Wrong error message: {e}")
            return False

    # Test 4: Case sensitivity (SHOULD FAIL - periods are lowercase)
    print("\n[TEST] Period case sensitivity")
    try:
        validate_period("1Y")  # Uppercase should fail
        print("  [FAIL] Should have raised ValueError for uppercase period '1Y'")
        return False
    except ValueError as e:
        if "invalid period" in str(e).lower() or "must be one of" in str(e).lower():
            print(f"  [OK] Correctly rejected uppercase: {e}")
        else:
            print(f"  [FAIL] Wrong error message: {e}")
            return False

    print("\n[OK] All period validation tests passed")
    return True


def test_ticker_validation():
    """Test ticker validation (existing functionality check)."""
    print("\n\nTesting ticker validation...")

    # Test 1: Valid tickers (SHOULD PASS)
    print("\n[TEST] Valid tickers should pass")
    try:
        validate_tickers(["AAPL", "MSFT", "GOOGL"])
        print("  [OK] Valid tickers accepted")
    except ValueError as e:
        print(f"  [FAIL] Valid tickers rejected: {e}")
        return False

    # Test 2: Empty list (SHOULD FAIL)
    print("\n[TEST] Empty ticker list should fail")
    try:
        validate_tickers([])
        print("  [FAIL] Should have raised ValueError for empty list")
        return False
    except ValueError as e:
        if "at least one" in str(e).lower():
            print(f"  [OK] Correctly rejected empty list: {e}")
        else:
            print(f"  [FAIL] Wrong error message: {e}")
            return False

    # Test 3: Too many tickers (SHOULD FAIL)
    print("\n[TEST] Too many tickers should fail")
    try:
        too_many = [f"TICK{i}" for i in range(25)]  # MAX is 20
        validate_tickers(too_many)
        print("  [FAIL] Should have raised ValueError for too many tickers")
        return False
    except ValueError as e:
        if "too many" in str(e).lower() or "maximum" in str(e).lower():
            print(f"  [OK] Correctly rejected too many tickers: {e}")
        else:
            print(f"  [FAIL] Wrong error message: {e}")
            return False

    print("\n[OK] All ticker validation tests passed")
    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("INPUT VALIDATION TESTS")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Weight Validation", test_weight_validation()))
    results.append(("Period Validation", test_period_validation()))
    results.append(("Ticker Validation", test_ticker_validation()))

    # Print summary
    print("\n\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n[OK] All validation tests passed!")
        print("\nKey improvements:")
        print("  - Negative weights are now caught and reported clearly")
        print("  - Weights > 1.0 are validated")
        print("  - Period validation happens in code, not just argparse")
        print("  - Better error messages for all validation failures")
        return 0
    else:
        print("\n[FAIL] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
