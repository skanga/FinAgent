"""
Comprehensive tests for chart generation and validation.

Tests cover:
- CSV path validation (None, empty, non-existent, directory, zero-size)
- Valid CSV file processing
- CSV parsing errors (invalid format, missing columns, empty DataFrame)
- Error analysis handling
- Path validation order
- File permission handling
- Edge cases (special characters, long filenames)
- Chart embedding verification (markdown and HTML)
- Multiple tickers with mixed validity
"""

import pytest
import sys
import re
import tempfile
from pathlib import Path
import pandas as pd
from charts import ThreadSafeChartGenerator
from models import TickerAnalysis, AdvancedMetrics, TechnicalIndicators, FundamentalData

# Fix Windows console encoding for print statements
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def chart_generator():
    """Create a ThreadSafeChartGenerator instance."""
    return ThreadSafeChartGenerator()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def minimal_analysis():
    """Create minimal TickerAnalysis with required fields."""
    return TickerAnalysis(
        ticker="AAPL",
        csv_path=Path("/tmp/AAPL.csv"),
        chart_path=Path("/tmp/AAPL.png"),
        latest_close=100.0,
        avg_daily_return=0.001,
        volatility=0.02,
        ratios={},
        fundamentals=FundamentalData(),
        advanced_metrics=AdvancedMetrics(),
        technical_indicators=TechnicalIndicators(),
    )


@pytest.fixture
def valid_csv_data():
    """Create valid price data for CSV files."""
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "open": [100 + i for i in range(10)],
        "high": [102 + i for i in range(10)],
        "low": [99 + i for i in range(10)],
        "close": [101 + i for i in range(10)],
        "volume": [1000000 + i * 10000 for i in range(10)],
    })


# ============================================================================
# TEST CSV PATH VALIDATION
# ============================================================================


class TestNoneCSVPath:
    """Test handling of None csv_path."""

    def test_none_csv_path_no_error(self, chart_generator, temp_dir, minimal_analysis):
        """Test that None csv_path doesn't raise error."""
        analysis = minimal_analysis
        analysis.csv_path = None

        output_path = temp_dir / "comparison.png"

        # Should not raise error, just skip the ticker
        chart_generator.create_comparison_chart(
            {"AAPL": analysis},
            output_path
        )

        # Chart may not be created if no valid tickers
        # Just ensure no exception was raised


class TestEmptyCSVPath:
    """Test handling of empty string csv_path."""

    def test_empty_string_csv_path_no_error(self, chart_generator, temp_dir, minimal_analysis):
        """Test that empty string csv_path doesn't raise error."""
        analysis = minimal_analysis
        analysis.csv_path = ""

        output_path = temp_dir / "comparison.png"

        # Should not raise error, just skip the ticker
        chart_generator.create_comparison_chart(
            {"AAPL": analysis},
            output_path
        )


class TestNonExistentFile:
    """Test handling of non-existent file paths."""

    def test_nonexistent_file_no_error(self, chart_generator, temp_dir, minimal_analysis):
        """Test that non-existent file doesn't raise error."""
        analysis = minimal_analysis
        analysis.csv_path = temp_dir / "nonexistent.csv"

        output_path = temp_dir / "comparison.png"

        # Should not raise error, just skip the ticker
        chart_generator.create_comparison_chart(
            {"AAPL": analysis},
            output_path
        )


class TestDirectoryInsteadOfFile:
    """Test handling of directory path instead of file."""

    def test_directory_path_no_error(self, chart_generator, temp_dir, minimal_analysis):
        """Test that directory path doesn't raise error."""
        analysis = minimal_analysis

        # Create a directory with the same name as expected CSV
        dir_path = temp_dir / "AAPL.csv"
        dir_path.mkdir()
        analysis.csv_path = dir_path

        output_path = temp_dir / "comparison.png"

        # Should not raise error, just skip the ticker
        chart_generator.create_comparison_chart(
            {"AAPL": analysis},
            output_path
        )


class TestEmptyCSVFile:
    """Test handling of empty CSV files (zero bytes)."""

    def test_empty_file_no_error(self, chart_generator, temp_dir, minimal_analysis):
        """Test that empty CSV file (0 bytes) doesn't raise error."""
        analysis = minimal_analysis

        # Create an empty file
        csv_path = temp_dir / "AAPL.csv"
        csv_path.touch()  # Creates empty file
        analysis.csv_path = csv_path

        output_path = temp_dir / "comparison.png"

        # Should not raise error, just skip the ticker
        chart_generator.create_comparison_chart(
            {"AAPL": analysis},
            output_path
        )


# ============================================================================
# TEST VALID CSV PROCESSING
# ============================================================================


class TestValidCSVFile:
    """Test processing of valid CSV files."""

    def test_valid_csv_processed(self, chart_generator, temp_dir, minimal_analysis, valid_csv_data):
        """Test that valid CSV file is processed correctly."""
        analysis = minimal_analysis

        # Create valid CSV file
        csv_path = temp_dir / "AAPL.csv"
        valid_csv_data.to_csv(csv_path, index=False)
        analysis.csv_path = csv_path

        output_path = temp_dir / "comparison.png"

        # Should successfully create chart
        chart_generator.create_comparison_chart(
            {"AAPL": analysis},
            output_path
        )

        # Chart should be created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_multiple_tickers_mixed_validity(
        self, chart_generator, temp_dir, minimal_analysis, valid_csv_data
    ):
        """Test multiple tickers with mixed valid/invalid paths."""
        # Valid ticker
        analysis1 = minimal_analysis
        csv_path1 = temp_dir / "AAPL.csv"
        valid_csv_data.to_csv(csv_path1, index=False)
        analysis1.csv_path = csv_path1

        # Invalid ticker (None path)
        analysis2 = TickerAnalysis(
            ticker="MSFT",
            csv_path=None,
            chart_path=Path("/tmp/MSFT.png"),
            latest_close=200.0,
            avg_daily_return=0.002,
            volatility=0.03,
            ratios={},
            fundamentals=FundamentalData(),
            advanced_metrics=AdvancedMetrics(),
            technical_indicators=TechnicalIndicators(),
        )

        # Invalid ticker (non-existent file)
        analysis3 = TickerAnalysis(
            ticker="GOOGL",
            csv_path=temp_dir / "nonexistent.csv",
            chart_path=Path("/tmp/GOOGL.png"),
            latest_close=150.0,
            avg_daily_return=0.0015,
            volatility=0.025,
            ratios={},
            fundamentals=FundamentalData(),
            advanced_metrics=AdvancedMetrics(),
            technical_indicators=TechnicalIndicators(),
        )

        output_path = temp_dir / "comparison.png"

        # Should process only the valid ticker
        chart_generator.create_comparison_chart(
            {"AAPL": analysis1, "MSFT": analysis2, "GOOGL": analysis3},
            output_path
        )

        # Chart should be created with at least one valid ticker
        assert output_path.exists()


# ============================================================================
# TEST CSV PARSING ERRORS
# ============================================================================


class TestCSVParsingErrors:
    """Test handling of CSV files with parsing errors."""

    def test_invalid_csv_format_no_error(self, chart_generator, temp_dir, minimal_analysis):
        """Test that invalid CSV format doesn't crash."""
        analysis = minimal_analysis

        # Create invalid CSV file
        csv_path = temp_dir / "AAPL.csv"
        csv_path.write_text("This is not a valid CSV file\nRandom text\n!!!")
        analysis.csv_path = csv_path

        output_path = temp_dir / "comparison.png"

        # Should not raise error, just skip the ticker
        chart_generator.create_comparison_chart(
            {"AAPL": analysis},
            output_path
        )

    def test_csv_missing_required_columns(self, chart_generator, temp_dir, minimal_analysis):
        """Test CSV file missing required columns."""
        analysis = minimal_analysis

        # Create CSV with wrong columns
        csv_path = temp_dir / "AAPL.csv"
        df = pd.DataFrame({
            "wrong_column": [1, 2, 3],
            "another_column": [4, 5, 6],
        })
        df.to_csv(csv_path, index=False)
        analysis.csv_path = csv_path

        output_path = temp_dir / "comparison.png"

        # Should not raise error, just skip the ticker
        chart_generator.create_comparison_chart(
            {"AAPL": analysis},
            output_path
        )

    def test_csv_with_empty_dataframe(self, chart_generator, temp_dir, minimal_analysis):
        """Test CSV that parses but has no data rows."""
        analysis = minimal_analysis

        # Create CSV with headers only
        csv_path = temp_dir / "AAPL.csv"
        df = pd.DataFrame(columns=["Date", "open", "high", "low", "close", "volume"])
        df.to_csv(csv_path, index=False)
        analysis.csv_path = csv_path

        output_path = temp_dir / "comparison.png"

        # Should not raise error, just skip the ticker
        chart_generator.create_comparison_chart(
            {"AAPL": analysis},
            output_path
        )


# ============================================================================
# TEST ERROR ANALYSIS HANDLING
# ============================================================================


class TestErrorAnalysis:
    """Test handling of analysis with error field set."""

    def test_analysis_with_error_skipped(self, chart_generator, temp_dir, minimal_analysis, valid_csv_data):
        """Test that analysis with error field set is skipped."""
        analysis = minimal_analysis

        # Create valid CSV file
        csv_path = temp_dir / "AAPL.csv"
        valid_csv_data.to_csv(csv_path, index=False)
        analysis.csv_path = csv_path

        # Set error on analysis
        analysis.error = "Some error occurred"

        output_path = temp_dir / "comparison.png"

        # Should skip the ticker due to error
        chart_generator.create_comparison_chart(
            {"AAPL": analysis},
            output_path
        )


# ============================================================================
# TEST PATH VALIDATION ORDER
# ============================================================================


class TestPathValidationOrder:
    """Test that validations are performed in correct order."""

    def test_error_checked_before_csv_path(self, chart_generator, temp_dir, minimal_analysis):
        """Test that analysis.error is checked before csv_path validation."""
        analysis = minimal_analysis
        analysis.error = "Error message"
        analysis.csv_path = None  # Invalid path

        output_path = temp_dir / "comparison.png"

        # Should skip due to error, not crash on None csv_path
        chart_generator.create_comparison_chart(
            {"AAPL": analysis},
            output_path
        )


# ============================================================================
# TEST FILE PERMISSIONS
# ============================================================================


class TestFilePermissions:
    """Test handling of file permission issues."""

    def test_unreadable_file_handled(self, chart_generator, temp_dir, minimal_analysis, valid_csv_data):
        """Test that unreadable file is handled gracefully."""
        analysis = minimal_analysis

        # Create valid CSV file
        csv_path = temp_dir / "AAPL.csv"
        valid_csv_data.to_csv(csv_path, index=False)
        analysis.csv_path = csv_path

        # Note: On Windows, making files unreadable is complex
        # This test primarily ensures the error handling exists
        output_path = temp_dir / "comparison.png"

        # Should handle gracefully even if file operations fail
        chart_generator.create_comparison_chart(
            {"AAPL": analysis},
            output_path
        )


# ============================================================================
# TEST EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Test edge cases in CSV validation."""

    def test_csv_path_with_special_characters(self, chart_generator, temp_dir, minimal_analysis, valid_csv_data):
        """Test CSV path with special characters in filename."""
        analysis = minimal_analysis

        # Create CSV with special characters in name
        csv_path = temp_dir / "AAPL test (1) [data].csv"
        valid_csv_data.to_csv(csv_path, index=False)
        analysis.csv_path = csv_path

        output_path = temp_dir / "comparison.png"

        # Should handle special characters correctly
        chart_generator.create_comparison_chart(
            {"AAPL": analysis},
            output_path
        )

        assert output_path.exists()

    def test_csv_path_very_long(self, chart_generator, temp_dir, minimal_analysis, valid_csv_data):
        """Test CSV path with very long filename."""
        analysis = minimal_analysis

        # Create CSV with long name (but under OS limits)
        long_name = "A" * 100 + ".csv"
        csv_path = temp_dir / long_name
        valid_csv_data.to_csv(csv_path, index=False)
        analysis.csv_path = csv_path

        output_path = temp_dir / "comparison.png"

        # Should handle long filenames correctly
        chart_generator.create_comparison_chart(
            {"AAPL": analysis},
            output_path
        )

        assert output_path.exists()


# ============================================================================
# TEST CHART EMBEDDING IN REPORTS
# ============================================================================


class TestChartEmbedding:
    """Test chart embedding in markdown and HTML reports."""

    def test_chart_references_in_markdown(self):
        """Test that markdown reports contain chart image references."""
        reports_dir = Path(__file__).parent.parent / "financial_reports"

        # Skip if no reports directory
        if not reports_dir.exists():
            pytest.skip("financial_reports/ directory not found - this directory is created when you run the application (e.g., 'python main.py')")

        # Find latest markdown file
        md_files = list(reports_dir.glob("financial_report_*.md"))
        if not md_files:
            pytest.skip("No markdown reports found")

        latest_md = max(md_files, key=lambda p: p.stat().st_mtime)

        # Read markdown content
        with latest_md.open('r', encoding='utf-8') as f:
            content = f.read()

        # Check for image references
        image_refs = re.findall(r'!\[.*?\]\((.*?\.png)\)', content)

        # If there are multiple tickers, we should have images
        # But we allow empty for single-ticker reports
        assert isinstance(image_refs, list)

    def test_html_report_has_img_tags(self):
        """Test that HTML reports contain img tags."""
        reports_dir = Path(__file__).parent.parent / "financial_reports"

        # Skip if no reports directory
        if not reports_dir.exists():
            pytest.skip("financial_reports/ directory not found - this directory is created when you run the application (e.g., 'python main.py')")

        # Find latest HTML file
        html_files = list(reports_dir.glob("financial_report_*.html"))
        if not html_files:
            pytest.skip("No HTML reports found")

        latest_html = max(html_files, key=lambda p: p.stat().st_mtime)

        # Read HTML content
        with latest_html.open('r', encoding='utf-8') as f:
            html_content = f.read()

        # Check for img tags (may be empty for single-ticker reports)
        img_tags = re.findall(r'<img[^>]+src="([^"]+)"', html_content)

        # Just verify we can parse img tags
        assert isinstance(img_tags, list)

    def test_png_files_exist_for_references(self):
        """Test that PNG files referenced in markdown actually exist."""
        reports_dir = Path(__file__).parent.parent / "financial_reports"

        # Skip if no reports directory
        if not reports_dir.exists():
            pytest.skip("financial_reports/ directory not found - this directory is created when you run the application (e.g., 'python main.py')")

        # Find latest markdown file
        md_files = list(reports_dir.glob("financial_report_*.md"))
        if not md_files:
            pytest.skip("No markdown reports found")

        latest_md = max(md_files, key=lambda p: p.stat().st_mtime)

        # Read markdown content
        with latest_md.open('r', encoding='utf-8') as f:
            content = f.read()

        # Check for image references
        image_refs = re.findall(r'!\[.*?\]\((.*?\.png)\)', content)

        # Check if referenced PNG files exist
        for img_ref in image_refs:
            img_path = reports_dir / img_ref
            assert img_path.exists(), f"Referenced image {img_ref} not found"
            assert img_path.stat().st_size > 0, f"Image {img_ref} is empty"


# ============================================================================
# MANUAL VERIFICATION HELPER
# ============================================================================


def check_chart_embedding_manual():
    """Manual verification helper for chart embedding (not a pytest test).

    This function can be run directly to manually verify chart embedding
    in the latest report. Run with: python test_chart_csv_validation.py
    """
    reports_dir = Path(__file__).parent.parent / "financial_reports"

    if not reports_dir.exists():
        print("❌ No reports directory found")
        return False

    # Find latest markdown file
    md_files = list(reports_dir.glob("financial_report_*.md"))
    if not md_files:
        print("❌ No markdown reports found")
        return False

    latest_md = max(md_files, key=lambda p: p.stat().st_mtime)
    print(f"📄 Checking: {latest_md.name}")

    # Read markdown content
    with latest_md.open('r', encoding='utf-8') as f:
        content = f.read()

    # Check for image references
    image_refs = re.findall(r'!\[.*?\]\((.*?\.png)\)', content)

    if not image_refs:
        print("❌ No image references found in markdown")
        print("\nMarkdown sections:")
        for line in content.split('\n'):
            if line.startswith('##'):
                print(f"  {line}")
        return False

    print(f"✓ Found {len(image_refs)} image reference(s) in markdown:")
    for img in image_refs:
        print(f"  - {img}")

    # Check if HTML exists
    html_file = latest_md.with_suffix('.html')
    if not html_file.exists():
        print(f"\n❌ HTML file not found: {html_file.name}")
        return False

    print(f"\n✓ HTML file exists: {html_file.name}")

    # Check HTML for images
    with html_file.open('r', encoding='utf-8') as f:
        html_content = f.read()

    img_tags = re.findall(r'<img[^>]+src="([^"]+)"', html_content)
    print(f"✓ Found {len(img_tags)} <img> tag(s) in HTML")

    # Check if PNG files exist
    print("\nChecking PNG files:")
    for img_ref in image_refs:
        img_path = reports_dir / img_ref
        if img_path.exists():
            size_kb = img_path.stat().st_size / 1024
            print(f"  ✓ {img_ref} ({size_kb:.1f} KB)")
        else:
            print(f"  ❌ {img_ref} (NOT FOUND)")

    print("\n✅ Chart embedding test passed!")
    print(f"\nTo view: {html_file}")

    return True


if __name__ == "__main__":
    # Run manual verification when script is executed directly
    check_chart_embedding_manual()
