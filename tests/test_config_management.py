"""
Unit tests for config.py module.

Tests configuration loading, environment variable handling,
provider detection, and validation.
"""

import pytest
import os
from unittest.mock import patch

from config import Config


class TestConfigInitialization:
    """Test Config initialization and environment variable loading."""

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-api-key-12345",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "BENCHMARK_TICKER": "SPY",
            "RISK_FREE_RATE": "0.03",
            "CACHE_TTL_HOURS": "48",
            "MAX_WORKERS": "5",
            "REQUEST_TIMEOUT": "60",
        },
        clear=True,
    )
    def test_config_loads_from_environment(self):
        """Test that configuration loads from environment variables."""
        config = Config()

        assert config.openai_api_key == "test-api-key-12345"
        assert config.openai_base_url == "https://api.openai.com/v1"
        assert config.benchmark_ticker == "SPY"
        assert config.risk_free_rate == 0.03
        assert config.cache_ttl_hours == 48
        assert config.max_workers == 5
        assert config.request_timeout == 60

    @patch.dict(os.environ, {}, clear=True)
    def test_config_uses_defaults_when_no_env_vars(self):
        """Test that configuration uses default values when env vars missing."""
        config = Config()

        # Should use defaults from constants
        assert config.benchmark_ticker == "SPY"  # DEFAULT_BENCHMARK_TICKER
        assert config.risk_free_rate == 0.02  # DEFAULT_RISK_FREE_RATE
        assert config.cache_ttl_hours == 24  # DEFAULT_CACHE_TTL_HOURS
        assert config.max_workers == 3  # DEFAULT_MAX_WORKERS
        assert config.request_timeout == 30  # DEFAULT_REQUEST_TIMEOUT

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}, clear=True)
    def test_config_requires_api_key(self):
        """Test that API key is required."""
        config = Config()

        assert config.openai_api_key == "sk-test123"
        assert config.openai_api_key is not None

    @patch.dict(os.environ, {}, clear=True)
    def test_config_api_key_defaults_to_none(self):
        """Test that missing API key defaults to None."""
        config = Config()

        assert config.openai_api_key is None


class TestProviderDetection:
    """Test LLM provider detection from API URL."""

    @patch.dict(
        os.environ, {"OPENAI_BASE_URL": "https://api.openai.com/v1"}, clear=True
    )
    def test_detect_openai_provider(self):
        """Test detection of OpenAI provider."""
        config = Config()

        assert config.provider == "openai"

    @patch.dict(
        os.environ, {"OPENAI_BASE_URL": "https://api.anthropic.com/v1"}, clear=True
    )
    def test_detect_anthropic_provider(self):
        """Test detection of Anthropic provider."""
        config = Config()

        assert config.provider == "anthropic"

    @patch.dict(
        os.environ,
        {"OPENAI_BASE_URL": "https://generativelanguage.googleapis.com/v1beta"},
        clear=True,
    )
    def test_detect_google_provider(self):
        """Test detection of Google provider."""
        config = Config()

        assert config.provider == "google"

    @patch.dict(
        os.environ, {"OPENAI_BASE_URL": "http://localhost:11434/v1"}, clear=True
    )
    def test_detect_ollama_provider(self):
        """Test detection of Ollama provider."""
        config = Config()

        assert config.provider == "ollama"

    @patch.dict(
        os.environ, {"OPENAI_BASE_URL": "https://some-custom-api.com/v1"}, clear=True
    )
    def test_detect_unknown_provider(self):
        """Test detection of unknown/custom provider."""
        config = Config()

        assert config.provider == "unknown"

    @patch.dict(os.environ, {}, clear=True)
    def test_provider_defaults_to_openai_when_no_url(self):
        """Test that provider defaults to openai when no URL specified."""
        config = Config()

        # When no base URL is specified, should default to OpenAI
        assert config.provider == "openai"


class TestNumericConversion:
    """Test conversion of string environment variables to numeric types."""

    @patch.dict(os.environ, {"RISK_FREE_RATE": "0.025"}, clear=True)
    def test_float_conversion_from_string(self):
        """Test conversion of string to float."""
        config = Config()

        assert isinstance(config.risk_free_rate, float)
        assert config.risk_free_rate == 0.025

    @patch.dict(os.environ, {"CACHE_TTL_HOURS": "72"}, clear=True)
    def test_int_conversion_from_string(self):
        """Test conversion of string to integer."""
        config = Config()

        assert isinstance(config.cache_ttl_hours, int)
        assert config.cache_ttl_hours == 72

    @patch.dict(os.environ, {"MAX_WORKERS": "10"}, clear=True)
    def test_max_workers_conversion(self):
        """Test max_workers conversion."""
        config = Config()

        assert isinstance(config.max_workers, int)
        assert config.max_workers == 10

    @patch.dict(os.environ, {"REQUEST_TIMEOUT": "120"}, clear=True)
    def test_timeout_conversion(self):
        """Test request_timeout conversion."""
        config = Config()

        assert isinstance(config.request_timeout, int)
        assert config.request_timeout == 120


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    @patch.dict(os.environ, {"RISK_FREE_RATE": "invalid_float"}, clear=True)
    def test_invalid_float_uses_default(self):
        """Test that invalid float value falls back to default."""
        config = Config()

        # Should use default value when conversion fails
        assert config.risk_free_rate == 0.02  # DEFAULT_RISK_FREE_RATE

    @patch.dict(os.environ, {"MAX_WORKERS": "not_a_number"}, clear=True)
    def test_invalid_int_uses_default(self):
        """Test that invalid int value falls back to default."""
        config = Config()

        # Should use default value when conversion fails
        assert config.max_workers == 3  # DEFAULT_MAX_WORKERS

    @patch.dict(os.environ, {"CACHE_TTL_HOURS": "0"}, clear=True)
    def test_zero_cache_ttl(self):
        """Test handling of zero cache TTL."""
        config = Config()

        assert config.cache_ttl_hours == 0

    @patch.dict(os.environ, {"CACHE_TTL_HOURS": "-1"}, clear=True)
    def test_negative_cache_ttl(self):
        """Test handling of negative cache TTL."""
        config = Config()

        # Negative values should be accepted (though may not make sense)
        assert config.cache_ttl_hours == -1

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "",  # Empty string
        },
        clear=True,
    )
    def test_empty_string_api_key(self):
        """Test handling of empty string API key."""
        # Empty strings should be rejected (validation now works correctly)
        with pytest.raises(ValueError, match="OPENAI_API_KEY cannot be empty string"):
            _ = Config()  # Should raise ValueError


class TestConfigValues:
    """Test specific configuration value behaviors."""

    @patch.dict(os.environ, {"BENCHMARK_TICKER": "QQQ"}, clear=True)
    def test_custom_benchmark_ticker(self):
        """Test setting custom benchmark ticker."""
        config = Config()

        assert config.benchmark_ticker == "QQQ"

    @patch.dict(os.environ, {"BENCHMARK_TICKER": "spy"}, clear=True)
    def test_lowercase_ticker_preserved(self):
        """Test that ticker case is preserved."""
        config = Config()

        # Tickers might be case-sensitive in some contexts
        assert config.benchmark_ticker == "spy"

    @patch.dict(
        os.environ,
        {
            "OPENAI_BASE_URL": "https://api.openai.com/v1/",  # Trailing slash
        },
        clear=True,
    )
    def test_base_url_with_trailing_slash(self):
        """Test base URL with trailing slash."""
        config = Config()

        # Should handle trailing slash
        assert "openai.com" in config.openai_base_url


class TestProviderDetectionEdgeCases:
    """Test provider detection with edge cases."""

    @patch.dict(
        os.environ,
        {"OPENAI_BASE_URL": "https://OPENAI.COM/v1"},  # Uppercase
        clear=True,
    )
    def test_provider_detection_case_insensitive(self):
        """Test that provider detection is case-insensitive."""
        config = Config()

        # Should detect OpenAI despite uppercase
        assert config.provider == "openai"

    @patch.dict(
        os.environ, {"OPENAI_BASE_URL": "https://api.openai.azure.com/v1"}, clear=True
    )
    def test_provider_detection_with_subdomain(self):
        """Test provider detection with subdomains."""
        config = Config()

        # Should detect OpenAI even with Azure subdomain
        assert config.provider == "openai"

    @patch.dict(os.environ, {"OPENAI_BASE_URL": ""}, clear=True)  # Empty string
    def test_provider_detection_with_empty_url(self):
        """Test provider detection with empty URL."""
        config = Config()

        # Should handle empty URL gracefully
        assert config.provider in ["openai", "unknown"]


class TestConfigAsContainer:
    """Test Config object as a container of settings."""

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-key",
            "BENCHMARK_TICKER": "SPY",
            "RISK_FREE_RATE": "0.02",
            "CACHE_TTL_HOURS": "24",
            "MAX_WORKERS": "3",
            "REQUEST_TIMEOUT": "30",
        },
        clear=True,
    )
    def test_config_has_all_expected_attributes(self):
        """Test that Config object has all expected attributes."""
        config = Config()

        # Check all expected attributes exist
        assert hasattr(config, "openai_api_key")
        assert hasattr(config, "openai_base_url")
        assert hasattr(config, "benchmark_ticker")
        assert hasattr(config, "risk_free_rate")
        assert hasattr(config, "cache_ttl_hours")
        assert hasattr(config, "max_workers")
        assert hasattr(config, "request_timeout")
        assert hasattr(config, "provider")

    @patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://api.openai.com/v1"},
        clear=True,
    )
    def test_config_values_are_accessible(self):
        """Test that all config values are accessible."""
        config = Config()

        # All values should be accessible without errors
        _ = config.openai_api_key
        _ = config.openai_base_url
        _ = config.benchmark_ticker
        _ = config.risk_free_rate
        _ = config.cache_ttl_hours
        _ = config.max_workers
        _ = config.request_timeout
        _ = config.provider


class TestMultipleConfigInstances:
    """Test behavior with multiple Config instances."""

    @patch.dict(os.environ, {"BENCHMARK_TICKER": "SPY"}, clear=True)
    def test_multiple_instances_use_same_env(self):
        """Test that multiple instances read from same environment."""
        config1 = Config()
        config2 = Config()

        assert config1.benchmark_ticker == config2.benchmark_ticker
        assert config1.risk_free_rate == config2.risk_free_rate

    @patch.dict(os.environ, {"BENCHMARK_TICKER": "SPY"}, clear=True)
    def test_env_change_affects_new_instance(self):
        """Test that environment changes affect new Config instances."""
        config1 = Config()
        assert config1.benchmark_ticker == "SPY"

        # Change environment
        os.environ["BENCHMARK_TICKER"] = "QQQ"

        config2 = Config()
        assert config2.benchmark_ticker == "QQQ"
        # Original instance should still have old value
        assert config1.benchmark_ticker == "SPY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
