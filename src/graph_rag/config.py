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
    # Провайдер-специфичные параметры (JSON), уезжают в extra_body запроса. Работают
    # только там, где провайдер их читает: cloud.ru — да, Ollama через /v1 — НЕТ, она
    # их молча игнорирует (проверено, см. llm_client._extra_body).
    llm_extra_options: str = ""
    # Штраф за повтор токенов — СТАНДАРТНЫЙ параметр OpenAI API, в отличие от
    # provider-specific repeat_penalty выше. Защита от зацикливания маленьких моделей
    # на temperature=0: без неё модель на шумном тексте уходит в повторяющуюся
    # генерацию, упирается в max_tokens и вызывает дорогой ретрай (на 20 Newsgroups
    # это давало ~2.6 LLM-вызова на документ вместо одного). 0.0 — выключено;
    # дефолт не меняет ключ дискового кэша, так что старые кэши остаются валидны.
    llm_frequency_penalty: float = 0.0

    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    # НЕ уезжает в подкаталог датасета (в отличие от остальных артефактов, см.
    # `dataset_artifacts_dir`): кэш ключуется по содержимому запроса (llm_client._cache_key),
    # поэтому одинаковые промпты переиспользуются между датасетами — общий кэш тут выгоден.
    llm_cache_dir: Path = PROJECT_ROOT / "artifacts" / "llm_cache"

    # Какой датасет собираем — ключ реестра `ingest.DATASET_LOADERS` ("bbc", "20newsgroups").
    # Переключается через DATASET=... в .env/окружении.
    dataset: str = "bbc"

    n_docs: int = 100
    random_seed: int = 42

    @property
    def dataset_artifacts_dir(self) -> Path:
        """Подкаталог артефактов текущего датасета (`artifacts/<dataset>/`).

        Разделение по датасетам обязательно: имена артефактов фиксированные
        (docs.parquet, embeddings.npy, graph.json...), и без подкаталога переключение
        DATASET молча затирало бы граф и эмбеддинги предыдущего датасета — причём
        частично, потому что pipeline пропускает шаги по наличию файла, так что
        docs.parquet от одного датасета мог бы сойтись с embeddings.npy от другого."""
        path = self.artifacts_dir / self.dataset
        path.mkdir(parents=True, exist_ok=True)
        return path

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
