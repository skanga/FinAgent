"""
Test concurrent processing with ThreadPoolExecutor.
"""

import pytest
import sys
import time
import logging
from pathlib import Path
from unittest.mock import Mock, patch
from threading import Lock, current_thread

# Configure stdout to handle unicode on Windows
# Skip this when running under pytest to avoid conflicts with pytest's capture
if sys.platform == "win32" and "pytest" not in sys.modules:
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class TestConcurrentProcessing:
    """Test concurrent processing with ThreadPoolExecutor."""

    def test_progress_tracker_thread_safety(self):
        """Test that ProgressTracker is thread-safe."""
        from utils import ProgressTracker
        from concurrent.futures import ThreadPoolExecutor

        progress = ProgressTracker(total=100, description="Thread Test")

        def update_progress(item_id):
            """Simulate concurrent updates"""
            time.sleep(0.01)  # Simulate work
            progress.update(f"Item-{item_id}", success=True)
            return item_id

        # Run 100 concurrent updates
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(update_progress, i) for i in range(100)]
            _results = [f.result() for f in futures]

        # Verify all updates completed
        assert progress.current == 100, f"Expected 100 updates, got {progress.current}"

    def test_concurrent_ticker_analysis(self):
        """Test concurrent ticker analysis in orchestrator."""
        from config import Config
        from orchestrator import FinancialReportOrchestrator
        from models import TickerAnalysis, AdvancedMetrics, TechnicalIndicators

        # Track which thread analyzed each ticker
        thread_tracker = {}
        tracker_lock = Lock()

        def mock_analyze_ticker(self, ticker, period, output_path, benchmark_returns, include_options=False, options_expirations=3):
            """Mock analyze_ticker that tracks thread usage"""
            thread_id = current_thread().name

            with tracker_lock:
                if thread_id not in thread_tracker:
                    thread_tracker[thread_id] = []
                thread_tracker[thread_id].append(ticker)

            # Simulate analysis work
            time.sleep(0.1)

            # Return mock analysis
            return TickerAnalysis(
                ticker=ticker,
                csv_path=f"/tmp/{ticker}.csv",
                chart_path=f"/tmp/{ticker}.png",
                latest_close=100.0,
                avg_daily_return=0.01,
                volatility=0.02,
                ratios={"pe_ratio": 15.0},
                fundamentals=None,
                advanced_metrics=AdvancedMetrics(sharpe_ratio=1.5, max_drawdown=-0.1),
                technical_indicators=TechnicalIndicators(rsi=50.0),
                sample_data=[],
                error=None,
            )

        # Create config
        config = Config(openai_api_key="test-key", max_workers=3)  # Use 3 workers

        # Create orchestrator
        with patch.object(FinancialReportOrchestrator, "__init__", lambda x, y: None):
            orchestrator = FinancialReportOrchestrator(config)
            orchestrator.config = config

            # Patch analyze_ticker method
            with patch.object(
                FinancialReportOrchestrator, "analyze_ticker", mock_analyze_ticker
            ):
                tickers = [f"TICK{i}" for i in range(10)]

                start = time.time()
                analyses = orchestrator._analyze_all_tickers(
                    tickers=tickers,
                    period="1y",
                    output_path=Path("/tmp"),
                    benchmark_returns=None,
                )
                elapsed = time.time() - start

        # Verify results
        assert len(analyses) == 10, f"Expected 10 analyses, got {len(analyses)}"

        # With 3 workers and 0.1s per ticker, should take ~0.4s (10 tickers / 3 workers * 0.1s)
        assert elapsed < 0.8, f"Parallel execution too slow: {elapsed:.2f}s"

        # Check thread distribution
        num_threads = len(thread_tracker)
        assert num_threads > 1, "Only 1 thread used (expected concurrent execution)"

    def test_concurrent_error_handling(self):
        """Test error handling in concurrent execution."""
        from config import Config
        from orchestrator import FinancialReportOrchestrator
        from models import TickerAnalysis
        from pathlib import Path

        def mock_analyze_ticker_with_errors(
            self, ticker, period, output_path, benchmark_returns, include_options=False, options_expirations=3
        ):
            """Mock that fails for some tickers"""
            time.sleep(0.05)

            # Fail on odd-numbered tickers
            if int(ticker[-1]) % 2 == 1:
                raise ValueError(f"Simulated error for {ticker}")

            # Return mock for even-numbered tickers
            return TickerAnalysis(
                ticker=ticker,
                csv_path=f"/tmp/{ticker}.csv",
                chart_path=f"/tmp/{ticker}.png",
                latest_close=100.0,
                avg_daily_return=0.01,
                volatility=0.02,
                ratios={},
                fundamentals=None,
                advanced_metrics=Mock(),
                technical_indicators=Mock(),
                sample_data=[],
                error=None,
            )

        config = Config(openai_api_key="test-key", max_workers=3)

        with patch.object(FinancialReportOrchestrator, "__init__", lambda x, y: None):
            orchestrator = FinancialReportOrchestrator(config)
            orchestrator.config = config

            with patch.object(
                FinancialReportOrchestrator,
                "analyze_ticker",
                mock_analyze_ticker_with_errors,
            ):
                tickers = [f"TICK{i}" for i in range(10)]

                analyses = orchestrator._analyze_all_tickers(
                    tickers=tickers,
                    period="1y",
                    output_path=Path("/tmp"),
                    benchmark_returns=None,
                )

        # Count successes and failures
        successful = [t for t, a in analyses.items() if not a.error]
        failed = [t for t, a in analyses.items() if a.error]

        # Should have 5 successes (even) and 5 failures (odd)
        assert len(successful) == 5, f"Expected 5 successes, got {len(successful)}"
        assert len(failed) == 5, f"Expected 5 failures, got {len(failed)}"

    def test_worker_limit(self):
        """Test that max_workers limit is respected."""
        from config import Config
        from orchestrator import FinancialReportOrchestrator
        from models import TickerAnalysis
        from pathlib import Path

        # Track concurrent executions
        max_concurrent = 0
        current_concurrent = 0
        semaphore_lock = Lock()

        def mock_analyze_with_tracking(
            self, ticker, period, output_path, benchmark_returns
        ):
            """Track max concurrent executions"""
            nonlocal max_concurrent, current_concurrent

            with semaphore_lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent

            time.sleep(0.1)  # Simulate work

            with semaphore_lock:
                current_concurrent -= 1

            return TickerAnalysis(
                ticker=ticker,
                csv_path="",
                chart_path="",
                latest_close=100.0,
                avg_daily_return=0.01,
                volatility=0.02,
                ratios={},
                fundamentals=None,
                advanced_metrics=Mock(),
                technical_indicators=Mock(),
                sample_data=[],
                error=None,
            )

        config = Config(openai_api_key="test-key", max_workers=2)

        with patch.object(FinancialReportOrchestrator, "__init__", lambda x, y: None):
            orchestrator = FinancialReportOrchestrator(config)
            orchestrator.config = config

            with patch.object(
                FinancialReportOrchestrator,
                "analyze_ticker",
                mock_analyze_with_tracking,
            ):
                tickers = [f"TICK{i}" for i in range(10)]

                _analyses = orchestrator._analyze_all_tickers(
                    tickers=tickers,
                    period="1y",
                    output_path=Path("/tmp"),
                    benchmark_returns=None,
                )

        assert (
            max_concurrent <= config.max_workers
        ), f"Exceeded worker limit ({max_concurrent} > {config.max_workers})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
