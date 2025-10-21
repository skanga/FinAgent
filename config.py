"""
Configuration management with validation.
"""
import os
import logging
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    # Look for .env file in current directory or parent directories
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        logger.debug(f"Loaded environment variables from {env_path.absolute()}")
    else:
        # Try to find .env in parent directory (useful when running from subdirectories)
        parent_env = Path(__file__).parent / '.env'
        if parent_env.exists():
            load_dotenv(dotenv_path=parent_env, override=False)
            logger.debug(f"Loaded environment variables from {parent_env.absolute()}")
except ImportError:
    logger.warning("python-dotenv not installed. Install it with: pip install python-dotenv")
except Exception as e:
    logger.warning(f"Could not load .env file: {e}")


@dataclass
class Config:
    """Application configuration with validation."""
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o"
    default_period: str = "1y"
    max_retries: int = 3
    request_timeout: int = 30
    cache_ttl_hours: int = 24
    max_workers: int = 3
    risk_free_rate: float = 0.02
    benchmark_ticker: str = "SPY"

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
            parsed = urlparse(self.openai_base_url)
            domain = parsed.netloc.lower()

            if 'openai.com' in domain:
                return "OpenAI"
            elif 'azure' in domain:
                return "Azure OpenAI"
            elif 'anthropic' in domain:
                return "Anthropic"
            elif 'localhost' in domain or '127.0.0.1' in domain:
                return "Local/Development"
            else:
                return f"Custom ({domain})"
        except Exception:
            return "Unknown"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY cannot be empty")
        
        if not (1 <= self.max_retries <= 10):
            raise ValueError("max_retries must be between 1 and 10")
        
        if not (10 <= self.request_timeout <= 300):
            raise ValueError("request_timeout must be between 10 and 300 seconds")
        
        if not (1 <= self.cache_ttl_hours <= 168):
            raise ValueError("cache_ttl_hours must be between 1 and 168")
        
        if not (1 <= self.max_workers <= 10):
            raise ValueError("max_workers must be between 1 and 10")
        
        if not (0 <= self.risk_free_rate <= 0.1):
            raise ValueError("risk_free_rate must be between 0 and 0.1")

        # Log configuration (with masked API key)
        provider = self._get_provider_from_url()
        masked_key = self._mask_api_key(self.openai_api_key)
        logger.info(f"Configuration loaded - Provider: {provider}, Model: {self.model_name}, API Key: {masked_key}")

    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        
        try:
            max_workers = int(os.getenv("MAX_WORKERS", "3"))
            cache_ttl = int(os.getenv("CACHE_TTL_HOURS", "24"))
        except ValueError as e:
            raise ValueError(f"Invalid integer in environment variables: {e}")
        
        return cls(
            openai_api_key=api_key,
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
            max_workers=max_workers,
            cache_ttl_hours=cache_ttl,
            benchmark_ticker=os.getenv("BENCHMARK_TICKER", "SPY")
        )
