import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    """Конфигурация подключения к базе данных"""

    host: str = os.getenv("DB_HOST", "localhost")
    database: str = os.getenv("DB_NAME", "postgres")
    user: str = os.getenv("DB_USER", "postgres")
    password: str | None = os.getenv("DB_PASSWORD")
    port: int = int(os.getenv("DB_PORT", "5432"))


@dataclass
class LlmConfig:
    """Конфигурация подключения к llm"""

    key: str | None = os.getenv("OPENROUTER_API_KEY")
    url: str = os.getenv("LLM_URL", "https://openrouter.ai/api/v1")
    # "qwen/qwen3-14b" "mistralai/ministral-8b" "z-ai/glm-4.5-air" "x-ai/grok-4-fast"
    default_model: str = os.getenv("LLM_DEFAULT_MODEL", "mistralai/ministral-8b")
    perplexity_model: str = os.getenv("LLM_PERPLEXITY_MODEL", "perplexity/sonar")


source_table_name = "person_source_data"
cleaned_table_name = "cleaned_person_source_data"
result_table_name = "person_result_data"
