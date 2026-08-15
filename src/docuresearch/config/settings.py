"""Application configuration, loaded from environment variables / .env.

All settings are optional at this layer - callers decide what to do when a
provider key is missing (mock/skip/disable), never this module.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from docuresearch.models.research import ResearchDepth


class ResearchDepthLimits(BaseSettings):
    """Source-count ranges per research depth. Overridable via env if needed later."""

    quick_min: int = 10
    quick_max: int = 20
    standard_min: int = 20
    standard_max: int = 40
    deep_min: int = 40
    deep_max: int = 80
    investigative_min: int = 80

    def target_for(self, depth: ResearchDepth, *, hard_cap: int) -> int:
        """Target source count for a depth, never exceeding the configured hard cap."""
        target = {
            ResearchDepth.QUICK: self.quick_max,
            ResearchDepth.STANDARD: self.standard_max,
            ResearchDepth.DEEP: self.deep_max,
            ResearchDepth.INVESTIGATIVE: self.investigative_min,
        }[depth]
        return min(target, hard_cap)


class Settings(BaseSettings):
    """Central application settings, sourced from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM providers (optional - missing ones are simply unavailable)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Observability
    langsmith_api_key: str | None = None
    langchain_tracing_v2: bool = False
    langchain_project: str = "docuresearch-ai"

    # Database
    database_url: str = "sqlite:///./data/docuresearch.db"

    # Filesystem
    cache_dir: str = "data/cache"

    # Research tool providers
    search_api_key: str | None = None
    news_api_key: str | None = None
    wikipedia_api_enabled: bool = True
    # Wikimedia's API rejects User-Agents that don't identify a contact URL/email
    # (https://meta.wikimedia.org/wiki/User-Agent_policy) - a bare descriptive
    # phrase like "research bot" is not enough, it must contain an actual URL.
    # Override with a real URL/mailto: for your project; the default satisfies
    # the policy without requiring configuration.
    wikipedia_contact: str = "https://github.com/docuresearch-ai/docuresearch-ai"

    # Research limits / performance
    max_research_sources: int = 40
    max_concurrent_requests: int = 5
    request_timeout: int = 30
    # Adaptive research loop (spec section 52): how many extra "go verify weak
    # claims harder" passes are allowed before moving on regardless.
    max_research_iterations: int = 2

    # Model defaults
    default_model: str = "gpt-4o-mini"
    default_temperature: float = 0.3

    # App behavior
    log_level: str = "INFO"
    environment: str = "development"

    depth_limits: ResearchDepthLimits = Field(default_factory=ResearchDepthLimits)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_any_llm_provider(self) -> bool:
        return self.has_openai or self.has_anthropic

    @property
    def has_search_provider(self) -> bool:
        return bool(self.search_api_key)

    @property
    def has_news_provider(self) -> bool:
        return bool(self.news_api_key)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton. Use `get_settings.cache_clear()` in tests."""
    return Settings()
