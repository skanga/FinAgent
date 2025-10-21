"""
Test Pydantic validation for request models.
"""
import sys
from pydantic import ValidationError
from models import TickerRequest, PortfolioRequest, NaturalLanguageRequest


def test_ticker_request_validation():
    """Test TickerRequest validation."""
    print("=" * 70)
    print("TICKER REQUEST VALIDATION TESTS")
    print("=" * 70)
    print()

    # Test 1: Valid ticker request
    print("[TEST] Valid ticker request")
    try:
        req = TickerRequest(ticker="aapl", period="1y")
        print(f"  [OK] Created: ticker={req.ticker}, period={req.period}")
        assert req.ticker == "AAPL", "Ticker should be uppercase"
    except ValidationError as e:
        print(f"  [FAIL] Unexpected error: {e}")
        return False

    # Test 2: Invalid ticker - too short
    print("\n[TEST] Empty ticker should fail")
    try:
        req = TickerRequest(ticker="", period="1y")
        print(f"  [FAIL] Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  [OK] Correctly rejected: {e.errors()[0]['msg']}")

    # Test 3: Invalid ticker - too long
    print("\n[TEST] Ticker too long should fail")
    try:
        req = TickerRequest(ticker="THISISTOOLONG", period="1y")
        print(f"  [FAIL] Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  [OK] Correctly rejected: {e.errors()[0]['msg']}")

    # Test 4: Invalid ticker - special characters
    print("\n[TEST] Invalid characters in ticker should fail")
    try:
        req = TickerRequest(ticker="AAP$L", period="1y")
        print(f"  [FAIL] Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  [OK] Correctly rejected: {str(e.errors()[0]['ctx']['error'])}")

    # Test 5: Suspicious ticker
    print("\n[TEST] Suspicious ticker name should fail")
    try:
        req = TickerRequest(ticker="test", period="1y")
        print(f"  [FAIL] Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  [OK] Correctly rejected: {str(e.errors()[0]['ctx']['error'])}")

    # Test 6: Invalid period
    print("\n[TEST] Invalid period should fail")
    try:
        req = TickerRequest(ticker="AAPL", period="1w")
        print(f"  [FAIL] Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  [OK] Correctly rejected: {str(e.errors()[0]['ctx']['error'])}")

    # Test 7: Valid period variations
    print("\n[TEST] All valid periods should work")
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    for period in valid_periods:
        try:
            req = TickerRequest(ticker="AAPL", period=period)
        except ValidationError as e:
            print(f"  [FAIL] Period '{period}' should be valid: {e}")
            return False
    print(f"  [OK] All {len(valid_periods)} valid periods accepted")

    # Test 8: Ticker normalization (lowercase to uppercase)
    print("\n[TEST] Ticker normalization")
    req = TickerRequest(ticker="msft", period="1y")
    assert req.ticker == "MSFT", "Ticker should be normalized to uppercase"
    print(f"  [OK] 'msft' normalized to '{req.ticker}'")

    print("\n[OK] All TickerRequest tests passed")
    return True


def test_portfolio_request_validation():
    """Test PortfolioRequest validation."""
    print("\n" + "=" * 70)
    print("PORTFOLIO REQUEST VALIDATION TESTS")
    print("=" * 70)
    print()

    # Test 1: Valid portfolio request without weights
    print("[TEST] Valid portfolio request without weights")
    try:
        req = PortfolioRequest(
            tickers=["AAPL", "MSFT", "GOOGL"],
            period="1y"
        )
        print(f"  [OK] Created with {len(req.tickers)} tickers")
        assert req.tickers == ["AAPL", "MSFT", "GOOGL"]
    except ValidationError as e:
        print(f"  [FAIL] Unexpected error: {e}")
        return False

    # Test 2: Valid portfolio request with weights
    print("\n[TEST] Valid portfolio request with weights")
    try:
        req = PortfolioRequest(
            tickers=["AAPL", "MSFT", "GOOGL"],
            period="1y",
            weights={"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}
        )
        print(f"  [OK] Created with weights: {req.weights}")
    except ValidationError as e:
        print(f"  [FAIL] Unexpected error: {e}")
        return False

    # Test 3: Empty ticker list
    print("\n[TEST] Empty ticker list should fail")
    try:
        req = PortfolioRequest(tickers=[], period="1y")
        print(f"  [FAIL] Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  [OK] Correctly rejected: {e.errors()[0]['msg']}")

    # Test 4: Too many tickers
    print("\n[TEST] Too many tickers should fail")
    try:
        many_tickers = [f"TICK{i}" for i in range(25)]  # MAX is 20
        req = PortfolioRequest(tickers=many_tickers, period="1y")
        print(f"  [FAIL] Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  [OK] Correctly rejected: {e.errors()[0]['msg']}")

    # Test 5: Ticker normalization
    print("\n[TEST] Ticker normalization in list")
    req = PortfolioRequest(
        tickers=["aapl", "msft", "googl"],
        period="1y"
    )
    assert req.tickers == ["AAPL", "MSFT", "GOOGL"]
    print(f"  [OK] Tickers normalized to uppercase: {req.tickers}")

    # Test 6: Weights don't sum to 1.0
    print("\n[TEST] Weights not summing to 1.0 should fail")
    try:
        req = PortfolioRequest(
            tickers=["AAPL", "MSFT", "GOOGL"],
            period="1y",
            weights={"AAPL": 0.3, "MSFT": 0.3, "GOOGL": 0.3}  # Sum = 0.9
        )
        print(f"  [FAIL] Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  [OK] Correctly rejected: {str(e.errors()[0]['ctx']['error'])}")

    # Test 7: Negative weights
    print("\n[TEST] Negative weights should fail")
    try:
        req = PortfolioRequest(
            tickers=["AAPL", "MSFT"],
            period="1y",
            weights={"AAPL": -0.3, "MSFT": 1.3}
        )
        print(f"  [FAIL] Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  [OK] Correctly rejected: {str(e.errors()[0]['ctx']['error'])}")

    # Test 8: Weights > 1.0
    print("\n[TEST] Weights exceeding 1.0 should fail")
    try:
        req = PortfolioRequest(
            tickers=["AAPL", "MSFT"],
            period="1y",
            weights={"AAPL": 0.5, "MSFT": 1.5}
        )
        print(f"  [FAIL] Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  [OK] Correctly rejected: {str(e.errors()[0]['ctx']['error'])}")

    # Test 9: Missing weight for ticker
    print("\n[TEST] Missing weight for ticker should fail")
    try:
        req = PortfolioRequest(
            tickers=["AAPL", "MSFT", "GOOGL"],
            period="1y",
            weights={"AAPL": 0.5, "MSFT": 0.5}  # Missing GOOGL
        )
        print(f"  [FAIL] Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  [OK] Correctly rejected: {str(e.errors()[0]['ctx']['error'])}")

    # Test 10: Extra weight for non-existent ticker
    print("\n[TEST] Extra weight for non-existent ticker should fail")
    try:
        req = PortfolioRequest(
            tickers=["AAPL", "MSFT"],
            period="1y",
            weights={"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}  # Extra GOOGL
        )
        print(f"  [FAIL] Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  [OK] Correctly rejected: {str(e.errors()[0]['ctx']['error'])}")

    # Test 11: Weights within tolerance
    print("\n[TEST] Weights within tolerance (0.999) should pass")
    try:
        req = PortfolioRequest(
            tickers=["AAPL", "MSFT", "GOOGL"],
            period="1y",
            weights={"AAPL": 0.333, "MSFT": 0.333, "GOOGL": 0.333}  # Sum = 0.999
        )
        print(f"  [OK] Weights accepted (sum={sum(req.weights.values()):.3f})")
    except ValidationError as e:
        print(f"  [FAIL] Should accept weights within tolerance: {e}")
        return False

    print("\n[OK] All PortfolioRequest tests passed")
    return True


def test_natural_language_request_validation():
    """Test NaturalLanguageRequest validation."""
    print("\n" + "=" * 70)
    print("NATURAL LANGUAGE REQUEST VALIDATION TESTS")
    print("=" * 70)
    print()

    # Test 1: Valid request
    print("[TEST] Valid natural language request")
    try:
        req = NaturalLanguageRequest(
            query="Compare AAPL and MSFT over the past year"
        )
        print(f"  [OK] Created: query='{req.query[:50]}...'")
    except ValidationError as e:
        print(f"  [FAIL] Unexpected error: {e}")
        return False

    # Test 2: Query too short
    print("\n[TEST] Query too short should fail")
    try:
        req = NaturalLanguageRequest(query="Hi")
        print(f"  [FAIL] Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  [OK] Correctly rejected: {e.errors()[0]['msg']}")

    # Test 3: Query too long
    print("\n[TEST] Query too long should fail")
    try:
        req = NaturalLanguageRequest(query="x" * 501)
        print(f"  [FAIL] Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  [OK] Correctly rejected: {e.errors()[0]['msg']}")

    # Test 4: Empty query (whitespace only)
    print("\n[TEST] Empty query (whitespace) should fail")
    try:
        req = NaturalLanguageRequest(query="     ")
        print(f"  [FAIL] Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  [OK] Correctly rejected: {str(e.errors()[0]['ctx']['error'])}")

    # Test 5: Query with leading/trailing whitespace
    print("\n[TEST] Query normalization (trim whitespace)")
    req = NaturalLanguageRequest(query="  Analyze AAPL  ")
    assert req.query == "Analyze AAPL"
    print(f"  [OK] Query trimmed: '{req.query}'")

    # Test 6: Custom output directory
    print("\n[TEST] Custom output directory")
    req = NaturalLanguageRequest(
        query="Analyze AAPL",
        output_dir="./custom_reports"
    )
    assert req.output_dir == "./custom_reports"
    print(f"  [OK] Custom output_dir: '{req.output_dir}'")

    print("\n[OK] All NaturalLanguageRequest tests passed")
    return True


def test_validation_error_messages():
    """Test that validation error messages are clear and helpful."""
    print("\n" + "=" * 70)
    print("VALIDATION ERROR MESSAGE QUALITY TESTS")
    print("=" * 70)
    print()

    print("[TEST] Clear error messages for common mistakes")

    # Test 1: Invalid ticker with clear message
    try:
        TickerRequest(ticker="AAP$L", period="1y")
    except ValidationError as e:
        error_msg = str(e.errors()[0]['ctx']['error'])
        assert "Invalid characters" in error_msg
        print(f"  [OK] Invalid ticker error: '{error_msg}'")

    # Test 2: Invalid period with suggestion
    try:
        TickerRequest(ticker="AAPL", period="1week")
    except ValidationError as e:
        error_msg = str(e.errors()[0]['ctx']['error'])
        assert "Must be one of" in error_msg
        print(f"  [OK] Invalid period error includes suggestions")

    # Test 3: Weight sum error shows actual sum
    try:
        PortfolioRequest(
            tickers=["AAPL", "MSFT"],
            weights={"AAPL": 0.3, "MSFT": 0.3}
        )
    except ValidationError as e:
        error_msg = str(e.errors()[0]['ctx']['error'])
        assert "got 0." in error_msg  # Shows actual sum
        print(f"  [OK] Weight sum error shows actual value")

    print("\n[OK] All error messages are clear and helpful")
    return True


def main():
    """Run all Pydantic validation tests."""
    print("=" * 70)
    print("PYDANTIC VALIDATION TEST SUITE")
    print("=" * 70)
    print()

    results = []

    # Run all test suites
    results.append(("TickerRequest", test_ticker_request_validation()))
    results.append(("PortfolioRequest", test_portfolio_request_validation()))
    results.append(("NaturalLanguageRequest", test_natural_language_request_validation()))
    results.append(("Error Messages", test_validation_error_messages()))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n[OK] All Pydantic validation tests passed!")
        print("\nKey features:")
        print("  - Automatic ticker normalization (uppercase)")
        print("  - Period validation against valid yfinance periods")
        print("  - Portfolio weight validation (sum, range, matching tickers)")
        print("  - Clear, actionable error messages")
        print("  - Field-level and model-level validation")
        return 0
    else:
        print("\n[FAIL] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
