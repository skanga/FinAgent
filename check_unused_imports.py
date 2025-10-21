"""
Check for unused imports in Python files.
"""
import ast
import os
import sys
from collections import defaultdict


class ImportChecker(ast.NodeVisitor):
    """Check for unused imports in a Python file."""

    def __init__(self, filename):
        self.filename = filename
        self.imports = {}  # name -> (module, line)
        self.used_names = set()
        self.current_scope = []

    def visit_Import(self, node):
        """Track 'import x' statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports[name] = (alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Track 'from x import y' statements."""
        for alias in node.names:
            if alias.name == '*':
                # Star imports - can't check usage reliably
                continue
            name = alias.asname if alias.asname else alias.name
            module = node.module or ''
            self.imports[name] = (f"{module}.{alias.name}", node.lineno)
        self.generic_visit(node)

    def visit_Name(self, node):
        """Track name usage."""
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Track attribute access."""
        if isinstance(node.value, ast.Name):
            self.used_names.add(node.value.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        """Track exception handler names."""
        if node.type:
            if isinstance(node.type, ast.Name):
                self.used_names.add(node.type.id)
            elif isinstance(node.type, ast.Tuple):
                for elt in node.type.elts:
                    if isinstance(elt, ast.Name):
                        self.used_names.add(elt.id)
        self.generic_visit(node)

    def check(self):
        """Return list of unused imports."""
        unused = []
        for name, (module, lineno) in self.imports.items():
            if name not in self.used_names:
                # Check for special cases that should be kept
                # Type checking imports
                if any(x in module.lower() for x in ['typing', 'optional', 'list', 'dict', 'tuple']):
                    continue
                # Re-exports (common in __init__.py)
                if self.filename.endswith('__init__.py'):
                    continue
                # Commonly used for side effects
                if module in ['dotenv.load_dotenv', 'logging.basicConfig']:
                    continue

                unused.append((name, module, lineno))
        return unused


def check_file(filepath):
    """Check a single file for unused imports."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=filepath)
        checker = ImportChecker(filepath)
        checker.visit(tree)

        return checker.check()
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return []
    except Exception as e:
        print(f"Error checking {filepath}: {e}")
        return []


def main():
    """Check all Python files for unused imports."""
    files_to_check = [
        'main.py', 'config.py', 'constants.py', 'cache.py', 'fetcher.py',
        'utils.py', 'alerts.py', 'analyzers.py', 'charts.py',
        'llm_interface.py', 'orchestrator.py', 'models.py'
    ]

    all_unused = defaultdict(list)

    for filename in files_to_check:
        if not os.path.exists(filename):
            continue

        unused = check_file(filename)
        if unused:
            all_unused[filename] = unused

    if all_unused:
        print("=" * 70)
        print("UNUSED IMPORTS FOUND")
        print("=" * 70)
        print()

        for filename, unused in sorted(all_unused.items()):
            print(f"{filename}:")
            for name, module, lineno in sorted(unused, key=lambda x: x[2]):
                print(f"  Line {lineno:3d}: {name:20s} (from {module})")
            print()

        total = sum(len(u) for u in all_unused.values())
        print(f"Total unused imports: {total}")
        return 1
    else:
        print("=" * 70)
        print("[OK] No unused imports found!")
        print("=" * 70)
        return 0


if __name__ == "__main__":
    sys.exit(main())
