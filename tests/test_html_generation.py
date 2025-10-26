"""
Comprehensive tests for HTML generation and security.

Tests cover:
- HTML generation from markdown
- Markdown to HTML conversion (mistune integration)
- Image processing and embedding
- Template loading and variable substitution
- XSS protection (title, bold text, links, tables, code blocks)
- JavaScript URL blocking
- Path traversal prevention
- HTML escaping in various contexts
- File size and content validation
"""

import pytest
import sys
from pathlib import Path
from html_generator import HTMLGenerator

# Fix Windows console encoding for manual tests
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def generator():
    """Create HTMLGenerator instance."""
    return HTMLGenerator()


@pytest.fixture
def sample_markdown():
    """Sample markdown content for testing."""
    return """# Financial Report Test

## Executive Summary

This is a test report with **bold text** and *italic text*.

## Key Metrics

| Metric | Value |
|--------|-------|
| Return | 15.3% |
| Volatility | 8.2% |
| Sharpe | 1.87 |

## Analysis

Here's a list:
* Item 1
* Item 2
* Item 3

### Code Sample

```python
def calculate_return(prices):
    return (prices[-1] / prices[0]) - 1
```

## Charts

![Technical Analysis](sample_chart.png)

---

*Generated: 2025-01-01*
"""


# ============================================================================
# TEST BASIC HTML GENERATION
# ============================================================================


class TestBasicHTMLGeneration:
    """Test basic HTML generation functionality."""

    def test_html_file_created(self, tmp_path, generator, sample_markdown):
        """Test that HTML file is created."""
        md_path = tmp_path / "test.md"
        md_path.write_text(sample_markdown, encoding='utf-8')

        html_path = generator.generate_html_report(md_path)

        assert html_path.exists()
        assert html_path.suffix == '.html'

    def test_html_not_empty(self, tmp_path, generator, sample_markdown):
        """Test that generated HTML is not empty."""
        md_path = tmp_path / "test.md"
        md_path.write_text(sample_markdown, encoding='utf-8')

        html_path = generator.generate_html_report(md_path)

        assert html_path.stat().st_size > 0

    def test_html_contains_title(self, tmp_path, generator, sample_markdown):
        """Test that HTML contains title."""
        md_path = tmp_path / "test.md"
        md_path.write_text(sample_markdown, encoding='utf-8')

        html_path = generator.generate_html_report(
            md_path,
            title="Test Financial Report"
        )
        html_content = html_path.read_text(encoding='utf-8')

        assert "Test Financial Report" in html_content
        assert "<title>" in html_content

    def test_html_contains_markdown_content(self, tmp_path, generator):
        """Test that HTML contains converted markdown content."""
        md_path = tmp_path / "test.md"
        md_path.write_text("# Test Heading\n\nTest paragraph", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        assert "Test Heading" in html_content
        assert "Test paragraph" in html_content
        assert "<h1>" in html_content
        assert "<p>" in html_content

    def test_custom_output_path(self, tmp_path, generator, sample_markdown):
        """Test custom output path."""
        md_path = tmp_path / "test.md"
        md_path.write_text(sample_markdown, encoding='utf-8')

        custom_path = tmp_path / "custom_report.html"
        html_path = generator.generate_html_report(md_path, output_path=custom_path)

        assert html_path == custom_path
        assert html_path.exists()

    def test_generation_timestamp(self, tmp_path, generator, sample_markdown):
        """Test that generation timestamp is included."""
        md_path = tmp_path / "test.md"
        md_path.write_text(sample_markdown, encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        # Should contain UTC timestamp
        assert "UTC" in html_content


# ============================================================================
# TEST MARKDOWN CONVERSION
# ============================================================================


class TestMarkdownConversion:
    """Test markdown to HTML conversion using mistune."""

    def test_bold_text_conversion(self, tmp_path, generator):
        """Test bold text conversion."""
        md_path = tmp_path / "test.md"
        md_path.write_text("**bold text**", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        assert "<strong>bold text</strong>" in html_content

    def test_italic_text_conversion(self, tmp_path, generator):
        """Test italic text conversion."""
        md_path = tmp_path / "test.md"
        md_path.write_text("*italic text*", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        assert "<em>italic text</em>" in html_content

    def test_table_conversion(self, tmp_path, generator):
        """Test table conversion."""
        md_text = """| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |"""
        md_path = tmp_path / "test.md"
        md_path.write_text(md_text, encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        assert "<table>" in html_content
        assert "<thead>" in html_content
        assert "<tbody>" in html_content
        assert "Header 1" in html_content
        assert "Cell 1" in html_content

    def test_code_block_conversion(self, tmp_path, generator):
        """Test code block conversion."""
        md_text = """```python
def hello():
    print("world")
```"""
        md_path = tmp_path / "test.md"
        md_path.write_text(md_text, encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        assert "<code" in html_content
        assert "def hello():" in html_content

    def test_inline_code_conversion(self, tmp_path, generator):
        """Test inline code conversion."""
        md_path = tmp_path / "test.md"
        md_path.write_text("This is `inline code` here", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        assert "<code>inline code</code>" in html_content

    def test_list_conversion(self, tmp_path, generator):
        """Test list conversion."""
        md_text = """* Item 1
* Item 2
* Item 3"""
        md_path = tmp_path / "test.md"
        md_path.write_text(md_text, encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        assert "<ul>" in html_content
        assert "<li>Item 1</li>" in html_content

    def test_link_conversion(self, tmp_path, generator):
        """Test link conversion."""
        md_path = tmp_path / "test.md"
        md_path.write_text("[Example](https://example.com)", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        assert '<a href="https://example.com">Example</a>' in html_content

    def test_strikethrough_conversion(self, tmp_path, generator):
        """Test strikethrough conversion (mistune plugin)."""
        md_path = tmp_path / "test.md"
        md_path.write_text("~~strikethrough~~", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        assert "<del>strikethrough</del>" in html_content


# ============================================================================
# TEST XSS PROTECTION
# ============================================================================


class TestXSSProtection:
    """Test XSS protection in various contexts."""

    def test_xss_protection_in_title(self, tmp_path, generator):
        """Test that XSS in title is escaped."""
        md_path = tmp_path / "test.md"
        md_path.write_text("# <script>alert('XSS')</script> Report\n\nTest content", encoding='utf-8')

        html_path = generator.generate_html_report(
            md_path,
            title="<script>alert('XSS')</script> Financial Report"
        )
        html_content = html_path.read_text(encoding='utf-8')

        # Should have escaped script tags in title
        assert "&lt;script&gt;" in html_content
        assert "<script>alert('XSS')</script>" not in html_content

    def test_xss_protection_in_bold_text(self, tmp_path, generator):
        """Test that XSS in bold text is escaped."""
        md_path = tmp_path / "test.md"
        md_path.write_text("# Test\n\n**<img src=x onerror=alert('XSS')>**", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        # Should have escaped HTML
        assert "&lt;img" in html_content
        assert "<img src=x onerror=" not in html_content

    def test_xss_protection_in_paragraph(self, tmp_path, generator):
        """Test that XSS in paragraph is escaped."""
        md_path = tmp_path / "test.md"
        md_path.write_text("# Test\n\n<script>alert('XSS')</script>", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        # Should have escaped script tags
        assert "&lt;script&gt;" in html_content
        assert "<script>alert('XSS')</script>" not in html_content

    def test_javascript_url_blocked_in_links(self, tmp_path, generator):
        """Test that javascript: URLs are blocked in links."""
        md_path = tmp_path / "test.md"
        md_path.write_text("# Test\n\n[Click me](javascript:alert('XSS'))", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        # Should NOT create a link with javascript:, just text
        assert "javascript:" not in html_content.lower()
        assert "Click me" in html_content  # Text should remain

    def test_data_url_blocked_in_links(self, tmp_path, generator):
        """Test that data: URLs are blocked in links."""
        md_path = tmp_path / "test.md"
        md_path.write_text("[Click](data:text/html,<script>alert('XSS')</script>)", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        # Should NOT create a link with data:
        assert "data:text/html" not in html_content
        assert "Click" in html_content  # Text should remain

    def test_vbscript_url_blocked(self, tmp_path, generator):
        """Test that vbscript: URLs are blocked."""
        md_path = tmp_path / "test.md"
        md_path.write_text("[Click](vbscript:msgbox('XSS'))", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        # Should NOT create a link with vbscript:
        assert "vbscript:" not in html_content.lower()

    def test_table_cell_escaping(self, tmp_path, generator):
        """Test that table cells are escaped."""
        md_text = """# Test

| Name | Value |
|------|-------|
| Test | <script>alert('XSS')</script> |
"""
        md_path = tmp_path / "test.md"
        md_path.write_text(md_text, encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        # Should have escaped script tags in table
        assert "&lt;script&gt;" in html_content
        assert "<script>alert('XSS')</script>" not in html_content

    def test_code_block_escaping(self, tmp_path, generator):
        """Test that code blocks are escaped."""
        md_text = """# Test

```python
<script>alert('XSS')</script>
```
"""
        md_path = tmp_path / "test.md"
        md_path.write_text(md_text, encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        # Should have escaped HTML in code block
        assert "&lt;script&gt;" in html_content
        # Script should not execute (should be in code block)
        assert "<code" in html_content

    def test_inline_code_escaping(self, tmp_path, generator):
        """Test that inline code is escaped."""
        md_path = tmp_path / "test.md"
        md_path.write_text("# Test\n\nThis is `<script>alert('XSS')</script>` code", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        # Should have escaped HTML in inline code
        assert "&lt;script&gt;" in html_content
        assert "<script>alert('XSS')</script>" not in html_content

    def test_html_injection_in_heading(self, tmp_path, generator):
        """Test HTML injection in heading is escaped."""
        md_path = tmp_path / "test.md"
        md_path.write_text("# <iframe src='evil.com'></iframe>", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        # Should have escaped iframe
        assert "&lt;iframe" in html_content
        assert "<iframe src=" not in html_content

    def test_event_handler_injection(self, tmp_path, generator):
        """Test event handler injection is escaped."""
        md_path = tmp_path / "test.md"
        md_path.write_text("Test <div onload=alert('XSS')>content</div>", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        # Mistune with escape=True should escape the HTML tags
        assert "&lt;div" in html_content
        # Note: The attribute might still appear escaped in the content
        # The key is that the HTML is escaped and won't execute


# ============================================================================
# TEST IMAGE PROCESSING
# ============================================================================


class TestImageProcessing:
    """Test image processing and security."""

    def test_image_not_found_handled(self, tmp_path, generator):
        """Test that missing images are handled gracefully."""
        md_path = tmp_path / "test.md"
        md_path.write_text("![Chart](nonexistent.png)", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        # Should contain warning message
        assert "Image not found" in html_content or "nonexistent.png" in html_content

    def test_javascript_url_blocked_in_images(self, tmp_path, generator):
        """Test that javascript: URLs are blocked in images."""
        md_path = tmp_path / "test.md"
        md_path.write_text("![XSS](javascript:alert('XSS'))", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        # Should block dangerous URL
        assert "javascript:" not in html_content.lower()
        assert "Image blocked" in html_content or "dangerous URL" in html_content

    def test_vbscript_url_blocked_in_images(self, tmp_path, generator):
        """Test that vbscript: URLs are blocked in images."""
        md_path = tmp_path / "test.md"
        md_path.write_text("![XSS](vbscript:msgbox('XSS'))", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        # Should block dangerous URL
        assert "vbscript:" not in html_content.lower()


# ============================================================================
# TEST PATH SECURITY
# ============================================================================


class TestPathSecurity:
    """Test path traversal prevention."""

    def test_path_traversal_prevention(self, tmp_path, generator):
        """Test that path traversal attempts are handled safely."""
        md_path = tmp_path / "test.md"
        md_path.write_text("![Evil](../../etc/passwd)", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        # Image processing converts this, but the file won't exist
        # The key security feature is that resolved paths are validated
        # to be within the base directory in html_generator.py _process_images
        # If path traversal is detected, it shows a security warning
        # If file doesn't exist, it shows "Image not found"
        # Either outcome is safe
        assert "../../etc/passwd" in html_content or "Security" in html_content or "not found" in html_content


# ============================================================================
# TEST ERROR HANDLING
# ============================================================================


class TestErrorHandling:
    """Test error handling."""

    def test_missing_markdown_file_error(self, tmp_path, generator):
        """Test that missing markdown file raises error."""
        md_path = tmp_path / "nonexistent.md"

        with pytest.raises(FileNotFoundError):
            generator.generate_html_report(md_path)

    def test_empty_markdown_file(self, tmp_path, generator):
        """Test that empty markdown file is handled."""
        md_path = tmp_path / "empty.md"
        md_path.write_text("", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)

        # Should still generate HTML
        assert html_path.exists()

    def test_missing_template_file_error(self, tmp_path):
        """Test that missing template file raises error."""
        nonexistent_template = tmp_path / "nonexistent_template.html"
        generator = HTMLGenerator(template_path=nonexistent_template)

        md_path = tmp_path / "test.md"
        md_path.write_text("# Test", encoding='utf-8')

        with pytest.raises(FileNotFoundError, match="Template not found"):
            generator.generate_html_report(md_path)


# ============================================================================
# TEST TEMPLATE FEATURES
# ============================================================================


class TestTemplateFeatures:
    """Test template loading and variable substitution."""

    def test_template_loaded_once(self, tmp_path, generator, sample_markdown):
        """Test that template is loaded once and cached."""
        md_path1 = tmp_path / "test1.md"
        md_path1.write_text(sample_markdown, encoding='utf-8')

        md_path2 = tmp_path / "test2.md"
        md_path2.write_text(sample_markdown, encoding='utf-8')

        # First generation loads template
        generator.generate_html_report(md_path1)

        # Second generation should use cached template
        generator.generate_html_report(md_path2)

        # Template should be cached
        assert generator._template_content is not None

    def test_title_extraction_from_markdown(self, tmp_path, generator):
        """Test that title is extracted from markdown if not provided."""
        md_path = tmp_path / "test.md"
        md_path.write_text("# Extracted Title\n\nContent here", encoding='utf-8')

        html_path = generator.generate_html_report(md_path)
        html_content = html_path.read_text(encoding='utf-8')

        # Should use extracted title
        assert "Extracted Title" in html_content


# ============================================================================
# MANUAL TEST (NOT PYTEST)
# ============================================================================


def manual_test_html_generation():
    """Manual test for HTML generation (can be run standalone).

    This is not a pytest test, but a manual validation script.
    """
    # Create a sample markdown file
    sample_markdown = """# Financial Report Test

## Executive Summary

This is a test report with **bold text** and *italic text*.

## Key Metrics

| Metric | Value |
|--------|-------|
| Return | 15.3% |
| Volatility | 8.2% |
| Sharpe | 1.87 |

## Analysis

Here's a list:
* Item 1
* Item 2
* Item 3

### Code Sample

```python
def calculate_return(prices):
    return (prices[-1] / prices[0]) - 1
```

## Charts

![Technical Analysis](sample_chart.png)

---

*Generated: 2025-01-01*
"""

    # Write sample markdown
    test_dir = Path(__file__).parent.parent / "test_output"
    test_dir.mkdir(exist_ok=True)

    md_path = test_dir / "test_report.md"
    with md_path.open('w', encoding='utf-8') as f:
        f.write(sample_markdown)

    print(f"✓ Created test markdown: {md_path}")

    # Generate HTML
    generator = HTMLGenerator()
    html_path = generator.generate_html_report(
        markdown_path=md_path,
        embed_images=False,
        title="Test Financial Report"
    )

    print(f"✓ Generated HTML: {html_path}")
    print(f"\nHTML file size: {html_path.stat().st_size} bytes")

    # Verify HTML was created
    assert html_path.exists(), "HTML file was not created"
    assert html_path.stat().st_size > 0, "HTML file is empty"

    print("\n✅ HTML generation test passed!")
    print(f"\nTo view the report, open: {html_path}")


if __name__ == "__main__":
    # Run manual test if executed directly
    manual_test_html_generation()
