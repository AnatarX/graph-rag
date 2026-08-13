"""Конфигурация проекта, загружается из .env / переменных окружения."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # Любой OpenAI-совместимый провайдер (cloud.ru Foundation Models, Ollama локально,
    # OpenRouter/Groq и т.п.) — переключается через LLM_BASE_URL/LLM_*_MODEL в .env.
    llm_api_key: str = ""
    llm_key_id: str = ""
    llm_key_secret: str = ""
    llm_iam_url: str = "https://iam.api.cloud.ru/api/v1/auth/token"
    llm_base_url: str = "https://foundation-models.api.cloud.ru/v1"
    llm_chat_model: str = "openai/gpt-oss-120b"
    llm_embed_model: str = ""
    # Доп. параметры запроса (JSON), прокидываются как extra_body в каждый вызов —
    # например, для локального Ollama на CPU полезно задать num_thread/num_ctx,
    # т.к. дефолтные настройки сильно недогружают многоядерный CPU.
    llm_extra_options: str = ""

    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    llm_cache_dir: Path = PROJECT_ROOT / "artifacts" / "llm_cache"

    n_docs: int = 100
    random_seed: int = 42

    def require_credentials(self) -> tuple[str, str]:
        if not self.llm_key_id or not self.llm_key_secret:
            raise RuntimeError(
                "Ни LLM_API_KEY, ни LLM_KEY_ID/LLM_KEY_SECRET не заданы. "
                "Скопируй .env.example в .env и заполни либо статический API-ключ "
                "(cloud.ru -> Foundation Models -> Create API Key, или ключ Ollama/OpenRouter/"
                "Groq), либо ключ сервисного аккаунта cloud.ru (Пользователи -> Сервисные "
                "аккаунты -> Access Credentials)."
            )
        return self.llm_key_id, self.llm_key_secret


settings = Settings()
