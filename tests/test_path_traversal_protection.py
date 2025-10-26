"""
Test path traversal protection in HTML generator.
"""

import pytest
from pathlib import Path
from html_generator import HTMLGenerator


def test_path_traversal_blocked(tmp_path):
    """Test that path traversal attempts are blocked."""
    generator = HTMLGenerator()

    # Create a markdown file
    md_path = tmp_path / "report.md"
    md_path.write_text("# Test\n\n![Secret](../../../etc/passwd)\n")

    # Generate HTML - should block the path traversal
    html_path = generator.generate_html_report(md_path)
    html_content = html_path.read_text(encoding='utf-8')

    # Should contain security warning, not the actual file content
    assert "Security: Image path outside allowed directory" in html_content
    assert "alert-danger" in html_content

    # Should NOT contain any file content
    assert "root:" not in html_content.lower()
    assert "bin:" not in html_content.lower()


def test_path_traversal_with_windows_style(tmp_path):
    """Test that path traversal using forward slashes is blocked (markdown syntax)."""
    generator = HTMLGenerator()

    # Create a markdown file with path traversal using forward slashes
    # Note: Markdown image syntax requires forward slashes, not backslashes
    md_path = tmp_path / "report.md"
    md_path.write_text("# Test\n\n![Secret](../../../Windows/System32/config/SAM)\n")

    # Generate HTML - should block the path traversal
    html_path = generator.generate_html_report(md_path)
    html_content = html_path.read_text(encoding='utf-8')

    # Should contain security warning
    assert "Security: Image path outside allowed directory" in html_content


def test_legitimate_subdirectory_allowed(tmp_path):
    """Test that legitimate subdirectory images are allowed."""
    generator = HTMLGenerator()

    # Create subdirectory with image
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "chart.png"

    # Create a minimal PNG file
    img_path.write_bytes(
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01'
        b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    )

    # Create markdown file referencing subdirectory image
    md_path = tmp_path / "report.md"
    md_path.write_text("# Test\n\n![Chart](images/chart.png)\n")

    # Generate HTML - should allow legitimate subdirectory access
    html_path = generator.generate_html_report(md_path)
    html_content = html_path.read_text(encoding='utf-8')

    # Should NOT contain security warning
    assert "Security: Image path outside allowed directory" not in html_content

    # Should contain the image reference
    assert "images/chart.png" in html_content or "chart.png" in html_content


def test_sibling_directory_blocked(tmp_path):
    """Test that sibling directory access is blocked."""
    generator = HTMLGenerator()

    # Create two sibling directories
    reports_dir = tmp_path / "reports"
    secrets_dir = tmp_path / "secrets"
    reports_dir.mkdir()
    secrets_dir.mkdir()

    # Create a secret file in secrets directory
    secret_file = secrets_dir / "password.txt"
    secret_file.write_text("super_secret_password")

    # Create markdown in reports directory trying to access sibling directory
    md_path = reports_dir / "report.md"
    md_path.write_text("# Test\n\n![Secret](../secrets/password.txt)\n")

    # Generate HTML - should block access to sibling directory
    html_path = generator.generate_html_report(md_path)
    html_content = html_path.read_text(encoding='utf-8')

    # Should contain security warning
    assert "Security: Image path outside allowed directory" in html_content

    # Should NOT contain secret content
    assert "super_secret_password" not in html_content


def test_absolute_path_blocked(tmp_path):
    """Test that absolute paths outside base directory are blocked."""
    generator = HTMLGenerator()

    # Create markdown file with absolute path
    md_path = tmp_path / "report.md"

    # Try to access a system file (will be different on Unix vs Windows)
    if Path("/etc/passwd").exists():
        dangerous_path = "/etc/passwd"
    else:
        dangerous_path = "C:\\Windows\\System32\\drivers\\etc\\hosts"

    md_path.write_text(f"# Test\n\n![Secret]({dangerous_path})\n")

    # Generate HTML - should block absolute path
    html_path = generator.generate_html_report(md_path)
    html_content = html_path.read_text(encoding='utf-8')

    # Should contain security warning
    assert "Security: Image path outside allowed directory" in html_content


def test_same_directory_allowed(tmp_path):
    """Test that images in same directory are allowed."""
    generator = HTMLGenerator()

    # Create image in same directory
    img_path = tmp_path / "chart.png"
    img_path.write_bytes(
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01'
        b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    )

    # Create markdown file in same directory
    md_path = tmp_path / "report.md"
    md_path.write_text("# Test\n\n![Chart](chart.png)\n")

    # Generate HTML - should allow same directory access
    html_path = generator.generate_html_report(md_path)
    html_content = html_path.read_text(encoding='utf-8')

    # Should NOT contain security warning
    assert "Security: Image path outside allowed directory" not in html_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
