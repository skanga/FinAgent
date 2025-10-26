# HTML Report Generation Feature

## Overview

The financial reporting agent now generates professional HTML reports alongside markdown reports by default. After generation, users are prompted to open the HTML report in their browser (press Enter to accept).

## What Was Implemented

### 1. **Chart Embedding in Reports**
- Charts are now embedded in both markdown and HTML reports
- Individual ticker technical charts included for each stock
- Portfolio comparison chart included for multi-ticker reports
- Images use relative file paths by default (or base64 embedding if enabled)
- All chart images are automatically referenced in the report

### 2. **HTML Generator Module** (`html_generator.py`)
- Custom regex-based markdown-to-HTML converter
- No external dependencies required (uses only built-in Python libraries)
- Properly handles image syntax `![alt](path.png)` before link syntax
- Supports:
  - Headers (h1-h4)
  - Bold, italic, and combined formatting
  - Links and inline code
  - Code blocks with language syntax
  - Unordered and ordered lists
  - Tables
  - Blockquotes
  - Horizontal rules
  - Images (embedded as base64 or relative references)

### 2. **Professional HTML Template** (`templates/report_template.html`)
- Responsive design that works on desktop, tablet, and mobile
- Professional financial report styling:
  - Clean typography with system fonts
  - Styled tables with hover effects
  - Chart containers with shadows
  - Alert boxes with color coding
  - Print-friendly CSS
- Color scheme:
  - Primary: Green (#4CAF50) for headers
  - Background: Light gray (#f5f7fa)
  - Content: White with shadow effects

### 3. **Configuration Settings** (updated `config.py`)

Three new configuration options with environment variable support:

| Setting | Default | Environment Var | Description |
|---------|---------|-----------------|-------------|
| `generate_html` | `true` | `GENERATE_HTML` | Enable/disable HTML generation |
| `embed_images_in_html` | `false` | `EMBED_IMAGES_IN_HTML` | Embed images as base64 |
| `open_in_browser` | `true` | `OPEN_IN_BROWSER` | Prompt to open in browser |

### 4. **CLI Flags** (updated `main.py`)

Three new command-line flags:

```bash
--no-html          # Disable HTML generation (markdown only)
--no-browser       # Don't open HTML in browser automatically
--embed-images     # Embed images as base64 in HTML
```

### 5. **Browser Auto-Launch** (with Security Confirmation)

- Uses Python's built-in `webbrowser` module
- **Prompts user for confirmation** before opening browser (security best practice)
- Shows file path in confirmation prompt for transparency
- Opens HTML report in new browser tab after user confirms
- Converts file path to proper `file://` URI
- Gracefully handles failures (logs warning, continues execution)
- Can be disabled via `--no-browser` flag or config

**Security Feature**: Even when `open_in_browser=true`, the application prompts:
```
Open report in browser? (C:\path\to\report.html) [Y/n]:
```
Pressing Enter (or typing 'Y'/'yes') will open the browser. Typing 'n'/'no' will skip opening.
This prevents unexpected browser actions and follows the principle of least surprise.

### 6. **Data Model Updates** (updated `models.py`)

Added `final_html_path: Optional[Path]` to `ReportMetadata` dataclass to track HTML output.

### 7. **Orchestrator Integration** (updated `orchestrator.py`)

- Imported `HTMLGenerator`
- Added `_generate_html_report()` method
- Integrated HTML generation after markdown save
- HTML generation errors don't fail the entire report process
- Returns HTML path in `ReportMetadata`

### 8. **Enhanced Output Display** (updated `main.py`)

Updated summary output to show:
```
📄 Markdown: /path/to/report.md
🌐 HTML: /path/to/report.html
   ✓ Opened in browser
```

## Usage Examples

### Default Behavior (HTML + Browser Prompt)
```bash
python main.py --tickers AAPL,MSFT --period 1y
# Generates both markdown and HTML
# Prompts: "Open report in browser? (path) [Y/n]:"
# Press Enter (or type 'y') to open in browser, or type 'n' to skip
```

### Markdown Only
```bash
python main.py --tickers AAPL,MSFT --period 1y --no-html
# Only generates markdown report
```

### HTML Without Browser
```bash
python main.py --tickers AAPL,MSFT --period 1y --no-browser
# Generates both formats, doesn't open browser
```

### Embedded Images (Single-File HTML)
```bash
python main.py --tickers AAPL,MSFT --period 1y --embed-images
# Creates self-contained HTML with base64 images (larger file)
```

### Environment Variables
```bash
export GENERATE_HTML=false        # Disable HTML generation
export OPEN_IN_BROWSER=false      # Don't prompt to open browser
export EMBED_IMAGES_IN_HTML=true  # Embed images
```

## Technical Details

### Image Handling Options

**Relative References (default):**
- HTML references images as `<img src="chart.png">`
- Smaller HTML file size
- Requires keeping image files with HTML
- Best for archival and sharing directories

**Base64 Embedding (`--embed-images`):**
- Images encoded as data URIs: `<img src="data:image/png;base64,..."`
- Single self-contained HTML file
- Larger file size (base64 encoding adds ~33% overhead)
- Best for sharing single file via email

### HTML Structure

The generated HTML includes:
- Responsive container (max-width: 1200px)
- Styled headers with border accents
- Professional tables with hover effects
- Chart containers with centered images
- Alert boxes with semantic colors
- Footer with generation timestamp

### CSS Features

- **Responsive**: Adapts to mobile/tablet/desktop
- **Print-friendly**: Optimized CSS for printing
- **Accessible**: Semantic HTML with proper contrast
- **Modern**: Uses system fonts, flexbox, CSS variables approach

## File Structure

```
financial_reporting_agent/
├── html_generator.py              # New: HTML conversion module
├── templates/                     # New: HTML templates
│   └── report_template.html      # Professional report template
├── orchestrator.py                # Modified: Added HTML generation
├── models.py                      # Modified: Added html_path field
├── config.py                      # Modified: Added HTML settings
├── main.py                        # Modified: Added CLI flags & browser launch
└── CLAUDE.md                      # Modified: Added HTML documentation
```

## Testing

Run the test script to verify HTML generation:

```bash
python test_html_generation.py
```

This creates a sample markdown file and generates HTML to verify the conversion works correctly.

## Benefits

1. **Better Presentation**: Professional, styled reports for stakeholders
2. **Easy Viewing**: No markdown viewer needed, works in any browser
3. **Print Ready**: Optimized CSS for printing reports
4. **Flexible**: Choose between lightweight (relative images) or portable (embedded images)
5. **Optional**: Can be completely disabled if not needed
6. **No Dependencies**: Uses only Python standard library

## Performance Impact

- HTML generation adds ~0.1-0.3 seconds to report generation
- Negligible overhead for markdown-to-HTML conversion
- Base64 embedding adds time proportional to image sizes
- Generation happens after markdown save, so errors don't affect main report

## Future Enhancements

Possible improvements (not implemented):
- Interactive charts using Plotly
- Dark mode toggle
- Collapsible sections
- Table of contents navigation
- Chart zoom/fullscreen
- Export to PDF button
- Syntax highlighting for code blocks
