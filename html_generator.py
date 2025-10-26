"""
HTML report generation from markdown with embedded images.

Note: This generator automatically converts ALL markdown content to HTML,
including options analysis sections. Options-specific charts (heatmaps, Greeks,
P&L diagrams, IV surfaces) are embedded as images just like stock charts.
"""

import re
import base64
import logging
import mistune
import html as html_lib  # Import as html_lib to avoid conflict with variable named 'html'
from pathlib import Path
from config import Config
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from mistune.renderers.html import HTMLRenderer

logger = logging.getLogger(__name__)


class SafeHTMLRenderer(HTMLRenderer):
    """Custom mistune renderer with additional security features."""

    def link(self, text: str, url: str, title: Optional[str] = None) -> str:
        """Render link with security validation to block dangerous URLs.

        Args:
            text: Link text
            url: Link URL
            title: Optional title attribute

        Returns:
            Safe HTML link or plain text if URL is dangerous
        """
        # Block javascript:, data:, and vbscript: URLs for XSS protection
        if url and url.strip().lower().startswith(('javascript:', 'data:', 'vbscript:')):
            logger.warning(f"Blocked dangerous URL in link: {url}")
            return html_lib.escape(text)  # Just return text, no link

        return super().link(text, url, title)

    def image(self, alt: str, url: str, title: Optional[str] = None) -> str:
        """Render image with security validation.

        Note: Image processing is actually handled by _process_images() later,
        but we block dangerous URLs here as well.

        Args:
            alt: Alt text
            url: Image URL
            title: Optional title attribute

        Returns:
            Safe HTML image tag
        """
        # Block javascript: and vbscript: URLs in image src
        if url and url.strip().lower().startswith(('javascript:', 'vbscript:')):
            logger.warning(f"Blocked dangerous URL in image: {url}")
            return '<p class="alert alert-warning">Image blocked: dangerous URL</p>'

        return super().image(alt, url, title)


class HTMLGenerator:
    """Converts markdown reports to HTML with styled templates and image handling."""

    def __init__(self, config: Optional[Config] = None, template_path: Optional[Path] = None):
        """
        Initialize HTML generator.

        Args:
            config: Configuration object containing AI model/provider info. Uses default if None.
            template_path: Path to HTML template file. Uses default if None.
        """
        if template_path is None:
            template_path = Path(__file__).parent / "templates" / "report_template.html"

        self.config = config if config is not None else Config.from_env()
        self.template_path = template_path
        self._template_content: Optional[str] = None

    def _load_template(self) -> str:
        """Load HTML template from file.

        Returns:
            Template content as string

        Raises:
            FileNotFoundError: If template file doesn't exist
        """
        if self._template_content is None:
            if not self.template_path.exists():
                raise FileNotFoundError(f"Template not found: {self.template_path}")

            with self.template_path.open("r", encoding="utf-8") as f:
                self._template_content = f.read()

        return self._template_content

    def _markdown_to_html(self, markdown_text: str) -> str:
        """
        Convert markdown to HTML using mistune library with security features.

        Args:
            markdown_text: Markdown content

        Returns:
            HTML content
        """
        # Create markdown parser with safe renderer and table support
        markdown = mistune.create_markdown(
            escape=True,  # Auto-escape HTML for XSS protection
            renderer=SafeHTMLRenderer(),
            plugins=['table', 'strikethrough', 'url']
        )

        # Convert markdown to HTML
        html = markdown(markdown_text)

        return html

    def _process_images(
        self,
        html: str,
        markdown_path: Path,
        embed_images: bool = False
    ) -> str:
        """
        Process image references in HTML.

        Args:
            html: HTML content with image references
            markdown_path: Path to original markdown file (for resolving relative paths)
            embed_images: If True, embed images as base64 data URIs

        Returns:
            HTML with processed image references
        """
        # Match HTML img tags (mistune already converted markdown to HTML)
        img_pattern = r'<img\s+src="([^"]+)"\s+alt="([^"]*)"\s*/>'

        def process_image(match):
            img_path_str = match.group(1)
            alt_text = match.group(2)  # Already HTML-escaped by mistune

            # URL-decode the path (mistune URL-encodes backslashes and other characters)
            import urllib.parse
            img_path_str = urllib.parse.unquote(img_path_str)

            # Resolve image path relative to markdown file
            if not img_path_str.startswith(('http://', 'https://', 'data:')):
                # Security: Check for absolute paths (Unix: starts with /, Windows: starts with drive letter)
                test_path = Path(img_path_str)
                if test_path.is_absolute():
                    logger.warning(f"Absolute path blocked: {img_path_str}")
                    return '<p class="alert alert-danger">Security: Image path outside allowed directory</p>'

                # Resolve paths to prevent path traversal attacks
                base_dir = markdown_path.parent.resolve()
                img_path = (markdown_path.parent / img_path_str).resolve()

                # Security: Validate that resolved path is within base directory
                # This prevents path traversal attacks like "../../etc/passwd"
                try:
                    img_path.relative_to(base_dir)
                except ValueError:
                    logger.warning(f"Path traversal attempt blocked: {img_path_str}")
                    return '<p class="alert alert-danger">Security: Image path outside allowed directory</p>'

                if not img_path.exists():
                    logger.warning(f"Image not found: {img_path}")
                    return f'<p class="alert alert-warning">Image not found: {html_lib.escape(img_path_str)}</p>'

                if embed_images:
                    # Embed as base64
                    try:
                        with img_path.open('rb') as f:
                            img_data = base64.b64encode(f.read()).decode('utf-8')

                        # Determine MIME type
                        ext = img_path.suffix.lower()
                        mime_types = {
                            '.png': 'image/png',
                            '.jpg': 'image/jpeg',
                            '.jpeg': 'image/jpeg',
                            '.gif': 'image/gif',
                            '.svg': 'image/svg+xml',
                        }
                        mime_type = mime_types.get(ext, 'image/png')

                        return f'<div class="chart-container"><img src="data:{mime_type};base64,{img_data}" alt="{alt_text}" /></div>'
                    except Exception as e:
                        logger.error(f"Failed to embed image {img_path}: {e}")
                        return f'<p class="alert alert-danger">Failed to embed image: {html_lib.escape(img_path_str)}</p>'
                else:
                    # Use relative path
                    try:
                        rel_path = img_path.relative_to(markdown_path.parent)
                        return f'<div class="chart-container"><img src="{html_lib.escape(str(rel_path))}" alt="{alt_text}" /></div>'
                    except ValueError:
                        # If relative path fails, use absolute path
                        return f'<div class="chart-container"><img src="{html_lib.escape(img_path.as_posix())}" alt="{alt_text}" /></div>'
            else:
                # External URL or data URI - wrap in div
                return f'<div class="chart-container"><img src="{img_path_str}" alt="{alt_text}" /></div>'

        return re.sub(img_pattern, process_image, html)

    def generate_html_report(
        self,
        markdown_path: Path,
        output_path: Optional[Path] = None,
        embed_images: bool = False,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Generate HTML report from markdown file.

        Args:
            markdown_path: Path to markdown file
            output_path: Path for output HTML file (default: same as markdown with .html extension)
            embed_images: If True, embed images as base64 data URIs
            title: Report title (default: extracted from markdown or filename)
            metadata: Additional metadata to include

        Returns:
            Path to generated HTML file

        Raises:
            FileNotFoundError: If markdown file doesn't exist
        """
        if not markdown_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

        # Read markdown content
        with markdown_path.open('r', encoding='utf-8') as f:
            markdown_content = f.read()

        # Extract title if not provided
        if title is None:
            title_match = re.search(r'^# (.+?)$', markdown_content, re.MULTILINE)
            if title_match:
                title = title_match.group(1)
            else:
                title = markdown_path.stem.replace('_', ' ').title()

        # Convert markdown to HTML
        logger.info("Converting markdown to HTML...")
        html_content = self._markdown_to_html(markdown_content)

        # Process images
        logger.info("Processing images...")
        html_content = self._process_images(html_content, markdown_path, embed_images)

        # Load template
        template = self._load_template()

        # Prepare template variables
        generation_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        ai_model = self.config.model_name
        ai_provider = self.config.provider

        # Replace template placeholders (escape all text for XSS protection)
        html_output = template.replace('{{ title }}', html_lib.escape(title))
        html_output = html_output.replace('{{ content }}', html_content)  # Already HTML
        html_output = html_output.replace('{{ generation_time }}', html_lib.escape(generation_time))
        html_output = html_output.replace('{{ ai_model }}', html_lib.escape(ai_model))
        html_output = html_output.replace('{{ ai_provider }}', html_lib.escape(ai_provider))

        # Determine output path
        if output_path is None:
            output_path = markdown_path.with_suffix('.html')

        # Write HTML file
        with output_path.open('w', encoding='utf-8') as f:
            f.write(html_output)

        logger.info(f"HTML report generated: {output_path}")
        return output_path
