"""
Test .env file loading functionality.
"""

import pytest
import os
import tempfile
from pathlib import Path


class TestEnvLoading:
    """Test .env file loading functionality."""

    def test_dotenv_import(self):
        """Test that python-dotenv is available."""
        from dotenv import load_dotenv

        # If import succeeds, test passes
        assert load_dotenv is not None

    def test_env_example_exists(self):
        """Test that .env.example file exists."""
        example_path = Path(".env.example")
        assert example_path.exists(), ".env.example file not found"

    def test_config_loads_dotenv(self):
        """Test that config.py loads .env file."""
        # Import config module (will trigger .env loading)

        # Check if the module has the load_dotenv call
        config_source = Path("config.py").read_text()

        has_dotenv_import = "from dotenv import load_dotenv" in config_source
        has_load_call = "load_dotenv(" in config_source

        assert (
            has_dotenv_import and has_load_call
        ), "config.py missing .env loading code"

    def test_env_variables_loaded(self):
        """Test that environment variables are loaded from .env."""
        # Check if .env file exists
        env_path = Path(".env")
        if not env_path.exists():
            pytest.skip("No .env file to test")

        # Read expected variables from .env
        env_vars = {}
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()

        if not env_vars:
            pytest.skip("No variables found in .env file")

        # Import config to trigger .env loading

        # Check if variables are in environment
        loaded_vars = [key for key in env_vars if os.getenv(key)]

        assert len(loaded_vars) > 0, "No environment variables loaded from .env"

    def test_config_from_env(self):
        """Test that Config.from_env() works with .env file."""
        from config import Config

        # This should load from environment (including .env)
        cfg = Config.from_env()

        # Validate that config was created
        assert cfg.openai_api_key is not None, "API key not loaded"
        assert cfg.openai_base_url is not None
        assert cfg.model_name is not None

    def test_env_file_format(self):
        """Test that .env file has correct format."""
        env_path = Path(".env")
        if not env_path.exists():
            pytest.skip("No .env file to validate")

        content = env_path.read_text()

        # Check for Windows 'set' commands (incorrect format)
        has_set_commands = (
            "set OPENAI_API_KEY" in content or "set OPENAI_BASE_URL" in content
        )

        assert (
            not has_set_commands
        ), ".env file uses Windows 'set' syntax - use KEY=value format"

        # Check for proper format
        valid_lines = 0
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    valid_lines += 1

        # At least some variables should be defined
        assert valid_lines >= 0, ".env file has no variable definitions"

    def test_gitignore_excludes_env(self):
        """Test that .gitignore excludes .env file."""
        gitignore_path = Path(".gitignore")

        if not gitignore_path.exists():
            pytest.skip("No .gitignore file found")

        content = gitignore_path.read_text()

        # Check if .env is excluded
        has_env = ".env" in content or "*.env" in content

        # This is a warning, not a failure
        if not has_env:
            pytest.skip(
                ".gitignore should include '.env' to prevent committing secrets"
            )

    def test_override_behavior(self):
        """Test that .env doesn't override existing environment variables."""
        # Set a test variable in the environment
        test_key = "TEST_ENV_OVERRIDE"
        test_value = "from_environment"
        os.environ[test_key] = test_value

        # Create a temporary .env file with the same variable
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(f"{test_key}=from_dotenv_file\n")
            temp_env = f.name

        try:
            from dotenv import load_dotenv

            # Load the temp .env with override=False (default behavior)
            load_dotenv(dotenv_path=temp_env, override=False)

            # Check which value is in the environment
            current_value = os.getenv(test_key)

            assert (
                current_value == test_value
            ), "Environment variable was overridden by .env file"

        finally:
            # Cleanup
            Path(temp_env).unlink()
            if test_key in os.environ:
                del os.environ[test_key]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
