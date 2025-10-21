"""
Test .env file loading functionality.
"""
import os
import sys
import tempfile
from pathlib import Path


def test_dotenv_import():
    """Test that python-dotenv is available."""
    print("Testing python-dotenv import...")

    try:
        from dotenv import load_dotenv
        print("  [PASS] python-dotenv imported successfully\n")
        return True
    except ImportError:
        print("  [FAIL] python-dotenv not installed")
        print("  Install with: pip install python-dotenv\n")
        return False


def test_env_file_exists():
    """Test that .env file exists."""
    print("Testing .env file existence...")

    env_path = Path('.env')
    exists = env_path.exists()

    if exists:
        print(f"  [PASS] .env file found at {env_path.absolute()}\n")
        return True
    else:
        print("  [WARN] .env file not found (optional)")
        print("  Copy .env.example to .env and configure your API key\n")
        return True  # Not a failure, just informational


def test_env_example_exists():
    """Test that .env.example file exists."""
    print("Testing .env.example file existence...")

    example_path = Path('.env.example')
    exists = example_path.exists()

    if exists:
        print(f"  [PASS] .env.example file found at {example_path.absolute()}\n")
        return True
    else:
        print("  [FAIL] .env.example file not found\n")
        return False


def test_config_loads_dotenv():
    """Test that config.py loads .env file."""
    print("Testing config.py .env loading...")

    try:
        # Import config module (will trigger .env loading)
        import config

        # Check if the module has the load_dotenv call
        config_source = Path('config.py').read_text()

        has_dotenv_import = 'from dotenv import load_dotenv' in config_source
        has_load_call = 'load_dotenv(' in config_source

        if has_dotenv_import and has_load_call:
            print("  [PASS] config.py properly loads .env file\n")
            return True
        else:
            print("  [FAIL] config.py missing .env loading code\n")
            return False

    except Exception as e:
        print(f"  [ERROR] Could not test config loading: {e}\n")
        return False


def test_env_variables_loaded():
    """Test that environment variables are loaded from .env."""
    print("Testing environment variables loaded from .env...")

    # Check if .env file exists
    env_path = Path('.env')
    if not env_path.exists():
        print("  [SKIP] No .env file to test\n")
        return True

    # Read expected variables from .env
    env_vars = {}
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()

    if not env_vars:
        print("  [WARN] No variables found in .env file\n")
        return True

    # Import config to trigger .env loading
    import config

    # Check if variables are in environment
    loaded_vars = []
    missing_vars = []

    for key in env_vars:
        if os.getenv(key):
            loaded_vars.append(key)
        else:
            missing_vars.append(key)

    print(f"  Loaded variables: {', '.join(loaded_vars) if loaded_vars else 'none'}")
    if missing_vars:
        print(f"  Missing variables: {', '.join(missing_vars)}")

    if loaded_vars:
        print("  [PASS] Environment variables loaded successfully\n")
        return True
    else:
        print("  [FAIL] No environment variables loaded\n")
        return False


def test_config_from_env():
    """Test that Config.from_env() works with .env file."""
    print("Testing Config.from_env() with .env file...")

    try:
        from config import Config

        # This should load from environment (including .env)
        cfg = Config.from_env()

        # Validate that config was created
        if cfg.openai_api_key:
            print(f"  API key loaded: {cfg._mask_api_key(cfg.openai_api_key)}")
            print(f"  Base URL: {cfg.openai_base_url}")
            print(f"  Model: {cfg.model_name}")
            print(f"  Provider: {cfg._get_provider_from_url()}")
            print("  [PASS] Config loaded successfully from environment\n")
            return True
        else:
            print("  [FAIL] API key not loaded\n")
            return False

    except ValueError as e:
        print(f"  [FAIL] Config validation error: {e}\n")
        return False
    except Exception as e:
        print(f"  [ERROR] Unexpected error: {e}\n")
        return False


def test_env_file_format():
    """Test that .env file has correct format."""
    print("Testing .env file format...")

    env_path = Path('.env')
    if not env_path.exists():
        print("  [SKIP] No .env file to validate\n")
        return True

    content = env_path.read_text()

    # Check for Windows 'set' commands (incorrect format)
    has_set_commands = 'set OPENAI_API_KEY' in content or 'set OPENAI_BASE_URL' in content

    if has_set_commands:
        print("  [FAIL] .env file uses Windows 'set' syntax")
        print("  Use KEY=value format instead (no 'set' prefix)\n")
        return False

    # Check for proper format
    valid_lines = 0
    for line in content.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            if '=' in line:
                valid_lines += 1
            else:
                print(f"  [WARN] Invalid line: {line}")

    if valid_lines > 0:
        print(f"  [PASS] .env file has proper format ({valid_lines} variables)\n")
        return True
    else:
        print("  [WARN] .env file has no variable definitions\n")
        return True


def test_gitignore_excludes_env():
    """Test that .gitignore excludes .env file."""
    print("Testing .gitignore excludes .env...")

    gitignore_path = Path('.gitignore')

    if not gitignore_path.exists():
        print("  [WARN] No .gitignore file found")
        print("  Create .gitignore to prevent committing secrets\n")
        return True

    content = gitignore_path.read_text()

    # Check if .env is excluded
    has_env = '.env' in content or '*.env' in content

    if has_env:
        print("  [PASS] .gitignore properly excludes .env files\n")
        return True
    else:
        print("  [WARN] .gitignore should include '.env' to prevent committing secrets\n")
        return True


def test_override_behavior():
    """Test that .env doesn't override existing environment variables."""
    print("Testing .env override behavior...")

    # Set a test variable in the environment
    test_key = "TEST_ENV_OVERRIDE"
    test_value = "from_environment"
    os.environ[test_key] = test_value

    # Create a temporary .env file with the same variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write(f"{test_key}=from_dotenv_file\n")
        temp_env = f.name

    try:
        from dotenv import load_dotenv

        # Load the temp .env with override=False (default behavior)
        load_dotenv(dotenv_path=temp_env, override=False)

        # Check which value is in the environment
        current_value = os.getenv(test_key)

        if current_value == test_value:
            print("  [PASS] Existing environment variables not overridden\n")
            return True
        else:
            print(f"  [FAIL] Environment variable was overridden: {current_value}\n")
            return False

    finally:
        # Cleanup
        Path(temp_env).unlink()
        if test_key in os.environ:
            del os.environ[test_key]


def main():
    """Run all .env loading tests."""
    print("=" * 60)
    print(".ENV FILE LOADING TESTS")
    print("=" * 60)
    print()

    tests = [
        test_dotenv_import,
        test_env_file_exists,
        test_env_example_exists,
        test_env_file_format,
        test_config_loads_dotenv,
        test_env_variables_loaded,
        test_config_from_env,
        test_gitignore_excludes_env,
        test_override_behavior
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  [ERROR] {test.__name__} failed:")
            print(f"    {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All .env loading tests passed!")
        print("\nYour application is properly configured to load environment")
        print("variables from .env files using python-dotenv.")
        return 0
    else:
        print(f"\n[PARTIAL] {total - passed} test(s) failed or skipped")
        return 1


if __name__ == "__main__":
    sys.exit(main())
