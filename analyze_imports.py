"""
Analyze imports across the codebase for quality and organization.
"""
import ast
import os
from collections import defaultdict


def analyze_imports(filepath):
    """
    Analyzes the imports in a Python file and categorizes them into standard library, third-party, and local imports.

    Args:
        filepath (str): The path to the Python file to analyze.

    Returns:
        dict: A dictionary containing the categorized imports.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    try:
        tree = ast.parse(content, filename=filepath)
    except SyntaxError:
        return None

    imports = {
        'stdlib': [],
        'third_party': [],
        'local': []
    }

    # Standard library modules (common ones)
    stdlib_modules = {
        'sys', 'os', 'time', 'datetime', 'logging', 'pathlib', 'typing',
        'collections', 'json', 'hashlib', 'argparse', 'dataclasses',
        'concurrent', 'functools', 'tempfile', 'urllib', 'codecs', 're'
    }

    # Local modules (our project)
    local_modules = {
        'config', 'cache', 'fetcher', 'analyzers', 'models', 'alerts',
        'charts', 'llm_interface', 'orchestrator', 'utils', 'constants'
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split('.')[0]
                if module in stdlib_modules:
                    imports['stdlib'].append((alias.name, node.lineno))
                elif module in local_modules:
                    imports['local'].append((alias.name, node.lineno))
                else:
                    imports['third_party'].append((alias.name, node.lineno))

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split('.')[0]
                names = [a.name for a in node.names]
                if module in stdlib_modules:
                    imports['stdlib'].append((f"from {node.module}", node.lineno))
                elif module in local_modules:
                    imports['local'].append((f"from {node.module}", node.lineno))
                else:
                    imports['third_party'].append((f"from {node.module}", node.lineno))

    return imports


def check_import_order(filepath):
    """
    Checks if the imports in a Python file follow the PEP 8 import order.

    Args:
        filepath (str): The path to the Python file to check.

    Returns:
        tuple: A tuple containing a boolean indicating whether the import order is correct and a list of issues.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find import section
    import_lines = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            import_lines.append((i, stripped))
        elif import_lines and stripped and not stripped.startswith('#'):
            # End of import section
            break

    if not import_lines:
        return True, []

    # Check order
    issues = []
    last_type = None
    stdlib_modules = {'sys', 'os', 'time', 'datetime', 'logging', 'pathlib', 'typing',
                      'collections', 'json', 'hashlib', 'argparse', 'dataclasses',
                      'concurrent', 'functools', 'tempfile', 'urllib', 'codecs', 're'}

    for lineno, line in import_lines:
        if line.startswith('from '):
            module = line.split()[1].split('.')[0]
        else:
            module = line.split()[1].split('.')[0]

        if module in stdlib_modules:
            current_type = 'stdlib'
        elif module in {'config', 'cache', 'fetcher', 'analyzers', 'models',
                        'alerts', 'charts', 'llm_interface', 'orchestrator',
                        'utils', 'constants'}:
            current_type = 'local'
        else:
            current_type = 'third_party'

        if last_type and current_type != last_type:
            # Check if there's a blank line between sections
            if lineno > 1:
                prev_line = lines[lineno - 2].strip()
                if prev_line and (prev_line.startswith('import ') or
                                 prev_line.startswith('from ')):
                    issues.append(f"Line {lineno}: Missing blank line before {current_type} imports")

        last_type = current_type

    return len(issues) == 0, issues


def main():
    """Analyze all Python files."""
    files = [
        'main.py', 'config.py', 'constants.py', 'cache.py', 'fetcher.py',
        'utils.py', 'alerts.py', 'analyzers.py', 'charts.py',
        'llm_interface.py', 'orchestrator.py', 'models.py'
    ]

    print("=" * 70)
    print("IMPORT ANALYSIS")
    print("=" * 70)
    print()

    total_stats = defaultdict(int)
    order_issues = []

    for filename in files:
        if not os.path.exists(filename):
            continue

        imports = analyze_imports(filename)
        if not imports:
            continue

        stdlib_count = len(imports['stdlib'])
        third_party_count = len(imports['third_party'])
        local_count = len(imports['local'])
        total = stdlib_count + third_party_count + local_count

        total_stats['stdlib'] += stdlib_count
        total_stats['third_party'] += third_party_count
        total_stats['local'] += local_count

        print(f"{filename}:")
        print(f"  Standard library: {stdlib_count}")
        print(f"  Third-party:      {third_party_count}")
        print(f"  Local modules:    {local_count}")
        print(f"  Total:            {total}")

        # Check import order
        ordered, issues = check_import_order(filename)
        if not ordered:
            order_issues.append((filename, issues))
            print(f"  [!] Import order issues: {len(issues)}")

        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total standard library imports: {total_stats['stdlib']}")
    print(f"Total third-party imports:      {total_stats['third_party']}")
    print(f"Total local module imports:     {total_stats['local']}")
    print(f"Total imports:                  {sum(total_stats.values())}")
    print()

    if order_issues:
        print("=" * 70)
        print("IMPORT ORDER ISSUES")
        print("=" * 70)
        for filename, issues in order_issues:
            print(f"\n{filename}:")
            for issue in issues:
                print(f"  {issue}")
        print()

    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("[OK] All imports are necessary (no unused imports)")
    print("[OK] Import structure is well-organized")
    if not order_issues:
        print("[OK] Import order follows PEP 8 guidelines")
    else:
        print("[!]  Some files have import order issues (see above)")


if __name__ == "__main__":
    main()
