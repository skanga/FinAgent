"""
Configuration management with validation.
"""

import os
import logging
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    # Look for .env file in current directory or parent directories
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        logger.debug(f"Loaded environment variables from {env_path.absolute()}")
    else:
        # Try to find .env in parent directory (useful when running from subdirectories)
        parent_env = Path(__file__).parent / ".env"
        if parent_env.exists():
            load_dotenv(dotenv_path=parent_env, override=False)
            logger.debug(f"Loaded environment variables from {parent_env.absolute()}")
except ImportError:
    logger.warning(
        "python-dotenv not installed. Install it with: pip install python-dotenv"
    )
except Exception as e:
    logger.warning(f"Could not load .env file: {e}")


class Config:
    """Application configuration with validation."""

    def __init__(
        self,
        openai_api_key: str = None,
        openai_base_url: str = None,
        model_name: str = None,
        default_period: str = None,
        max_retries: int = None,
        request_timeout: int = None,
        cache_ttl_hours: int = None,
        max_workers: int = None,
        risk_free_rate: float = None,
        benchmark_ticker: str = None,
        generate_html: bool = None,
        embed_images_in_html: bool = None,
        open_in_browser: bool = None,
        # Options-related parameters
        include_options: bool = None,
        options_cache_ttl_hours: int = None,
        options_expirations: int = None,
    ):
        """Initialize configuration, loading from environment variables if not specified."""
        # Load from environment variables with defaults
        from constants import Defaults

        self.openai_api_key = (
            openai_api_key
            if openai_api_key is not None
            else os.getenv("OPENAI_API_KEY")
        )
        self.openai_base_url = (
            openai_base_url
            if openai_base_url is not None
            else os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        self.model_name = (
            model_name
            if model_name is not None
            else os.getenv("OPENAI_MODEL", "gpt-4o")
        )
        self.default_period = default_period if default_period is not None else "1y"
        self.max_retries = max_retries if max_retries is not None else 3

        # Load numeric values from environment with proper type conversion
        if request_timeout is not None:
            self.request_timeout = request_timeout
        else:
            try:
                self.request_timeout = int(
                    os.getenv("REQUEST_TIMEOUT", str(Defaults.REQUEST_TIMEOUT))
                )
            except ValueError:
                self.request_timeout = Defaults.REQUEST_TIMEOUT

        if cache_ttl_hours is not None:
            self.cache_ttl_hours = cache_ttl_hours
        else:
            try:
                self.cache_ttl_hours = int(
                    os.getenv("CACHE_TTL_HOURS", str(Defaults.CACHE_TTL_HOURS))
                )
            except ValueError:
                self.cache_ttl_hours = Defaults.CACHE_TTL_HOURS

        if max_workers is not None:
            self.max_workers = max_workers
        else:
            try:
                self.max_workers = int(
                    os.getenv("MAX_WORKERS", str(Defaults.MAX_WORKERS))
                )
            except ValueError:
                self.max_workers = Defaults.MAX_WORKERS

        if risk_free_rate is not None:
            self.risk_free_rate = risk_free_rate
        else:
            try:
                self.risk_free_rate = float(
                    os.getenv("RISK_FREE_RATE", str(Defaults.RISK_FREE_RATE))
                )
            except ValueError:
                self.risk_free_rate = Defaults.RISK_FREE_RATE

        self.benchmark_ticker = (
            benchmark_ticker
            if benchmark_ticker is not None
            else os.getenv("BENCHMARK_TICKER", Defaults.BENCHMARK_TICKER)
        )

        # HTML generation settings
        if generate_html is not None:
            self.generate_html = generate_html
        else:
            env_val = os.getenv("GENERATE_HTML", "true").lower()
            self.generate_html = env_val in ("true", "1", "yes", "on")

        if embed_images_in_html is not None:
            self.embed_images_in_html = embed_images_in_html
        else:
            env_val = os.getenv("EMBED_IMAGES_IN_HTML", "false").lower()
            self.embed_images_in_html = env_val in ("true", "1", "yes", "on")

        if open_in_browser is not None:
            self.open_in_browser = open_in_browser
        else:
            env_val = os.getenv("OPEN_IN_BROWSER", "true").lower()
            self.open_in_browser = env_val in ("true", "1", "yes", "on")

        # Options analysis settings
        if include_options is not None:
            self.include_options = include_options
        else:
            env_val = os.getenv("INCLUDE_OPTIONS", "false").lower()
            self.include_options = env_val in ("true", "1", "yes", "on")

        if options_cache_ttl_hours is not None:
            self.options_cache_ttl_hours = options_cache_ttl_hours
        else:
            try:
                self.options_cache_ttl_hours = int(
                    os.getenv("OPTIONS_CACHE_TTL_HOURS", str(Defaults.OPTIONS_CACHE_TTL_HOURS))
                )
            except ValueError:
                self.options_cache_ttl_hours = Defaults.OPTIONS_CACHE_TTL_HOURS

        if options_expirations is not None:
            self.options_expirations = options_expirations
        else:
            try:
                self.options_expirations = int(
                    os.getenv("OPTIONS_EXPIRATIONS", str(Defaults.DEFAULT_OPTIONS_EXPIRATIONS))
                )
            except ValueError:
                self.options_expirations = Defaults.DEFAULT_OPTIONS_EXPIRATIONS

        # Computed field
        self.provider = None

        # Call post-init validation
        self.__post_init__()

    def __repr__(self) -> str:
        """Safe string representation that masks API key."""
        masked_key = self._mask_api_key(self.openai_api_key)
        return (
            f"Config(openai_api_key='{masked_key}', "
            f"openai_base_url='{self.openai_base_url}', "
            f"model_name='{self.model_name}', ...)"
        )

    def __str__(self) -> str:
        """Safe string conversion that masks API key."""
        return self.__repr__()

    @staticmethod
    def _mask_api_key(api_key: str) -> str:
        """Mask API key for safe logging/display."""
        if not api_key:
            return "NOT_SET"
        if len(api_key) <= 8:
            return "***"
        # Show first 4 and last 4 characters
        return f"{api_key[:4]}...{api_key[-4:]}"

    def _get_provider_from_url(self) -> str:
        """Extract provider name from base URL."""
        try:
            if not self.openai_base_url:
                return "openai"

            parsed = urlparse(self.openai_base_url)
            domain = parsed.netloc.lower()

            # Use match/case for cleaner provider detection
            match domain:
                case d if "openai" in d:  # Matches openai.com, api.openai.azure.com, etc.
                    return "openai"
                case d if "anthropic" in d:
                    return "anthropic"
                case d if "groq" in d:
                    return "groq"
                case d if "cerebras" in d:
                    return "cerebras"
                case d if "sambanova" in d:
                    return "sambanova"
                case d if "inception" in d:
                    return "inception"
                case d if "openrouter" in d:
                    return "openrouter"
                case d if "gemini" in d:
                    return "google"
                case d if "generativelanguage.googleapis.com" in d:
                    return "google"
                case d if "localhost" in d or "127.0.0.1" in d:
                    # Check for Ollama default port
                    return "ollama" if "11434" in self.openai_base_url else "local"
                case _:
                    return "unknown"
        except Exception:
            return "openai"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Compute provider from URL
        self.provider = self._get_provider_from_url()

        # Validate API key is not empty string (but allow None for testing)
        if self.openai_api_key is not None and self.openai_api_key == "":
            raise ValueError("OPENAI_API_KEY cannot be empty string")

        # Validate base URL is a valid URL
        if self.openai_base_url:
            try:
                parsed = urlparse(self.openai_base_url)
                if not parsed.scheme or not parsed.netloc:
                    raise ValueError(
                        f"OPENAI_BASE_URL must be a valid URL with scheme and host. Got: {self.openai_base_url}"
                    )
                if parsed.scheme not in ("http", "https"):
                    raise ValueError(
                        f"OPENAI_BASE_URL must use http or https scheme. Got: {parsed.scheme}"
                    )
            except Exception as e:
                if isinstance(e, ValueError):
                    raise
                raise ValueError(f"Invalid OPENAI_BASE_URL: {e}") from e

        # Allow wider ranges for testing purposes
        if not (1 <= self.max_retries <= 10):
            logger.warning(
                f"max_retries {self.max_retries} outside recommended range [1, 10]"
            )

        if not (10 <= self.request_timeout <= 300):
            logger.warning(
                f"request_timeout {self.request_timeout} outside recommended range [10, 300]"
            )

        if not (0 <= self.cache_ttl_hours <= 168):
            logger.warning(
                f"cache_ttl_hours {self.cache_ttl_hours} outside recommended range [0, 168]"
            )

        if not (1 <= self.max_workers <= 10):
            logger.warning(
                f"max_workers {self.max_workers} outside recommended range [1, 10]"
            )

        if not (0 <= self.risk_free_rate <= 0.1):
            logger.warning(
                f"risk_free_rate {self.risk_free_rate} outside recommended range [0, 0.1]"
            )

        # Log configuration (with masked API key) only if API key is set
        if self.openai_api_key:
            masked_key = self._mask_api_key(self.openai_api_key)
            logger.info(
                f"Configuration loaded - Provider: {self.provider}, Model: {self.model_name}, API Key: {masked_key}"
            )

    @classmethod
    def from_env(cls) -> "Config":
        """
        Loads the configuration from environment variables.

        This is a convenience method that validates the API key is set,
        then delegates to __init__() which handles all environment variable
        loading and validation.

        Returns:
            Config: The configuration object.

        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")

        # Simply call __init__() with no arguments - it will load everything from env vars
        # This eliminates code duplication since __init__() already handles all env var loading
        return cls()
