"""Тонкая обёртка над OpenAI-совместимым LLM API (по умолчанию cloud.ru Foundation Models).

Используем официальный `openai` SDK с переопределённым `base_url`, поэтому код
портируем на любой OpenAI-совместимый провайдер (cloud.ru, локальный Ollama,
OpenRouter, Groq и т.п.) — достаточно поменять LLM_BASE_URL/LLM_API_KEY/LLM_*_MODEL
в .env, без правок кода.

Все вызовы кэшируются на диске по хэшу запроса: и эмбеддинги, и chat-completions
дергаются многократно на одном и том же корпусе при итеративной разработке пайплайна,
а платный API-вызов — дорогая операция, которую не хочется повторять просто потому,
что перезапустили скрипт.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx
import numpy as np
import typer
from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from graph_rag.config import settings

_RETRYABLE = (APIConnectionError, RateLimitError, APIError)


def _extra_body() -> dict | None:
    if not settings.llm_extra_options:
        return None
    return {"options": json.loads(settings.llm_extra_options)}

# cloud.ru не принимает ключ сервисного аккаунта напрямую в chat/completions — его нужно
# сначала обменять на короткоживущий IAM access-token (см. cloud.ru IAM API), которым уже
# и авторизуются запросы к Foundation Models. Токен живёт 1 час, поэтому кэшируем его в
# памяти процесса и обновляем заранее, за минуту до истечения.
_token_cache: dict[str, float | str] = {}
_TOKEN_REFRESH_MARGIN_SEC = 60


@retry(
    retry=retry_if_exception_type((httpx.TransportError,)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _fetch_iam_token(key_id: str, key_secret: str) -> tuple[str, float]:
    response = httpx.post(
        settings.llm_iam_url,
        json={"keyId": key_id, "secret": key_secret},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["access_token"], time.time() + data["expires_in"]


def get_access_token() -> str:
    token = _token_cache.get("token")
    expires_at = _token_cache.get("expires_at", 0.0)
    if token and time.time() < expires_at - _TOKEN_REFRESH_MARGIN_SEC:
        return token  # type: ignore[return-value]

    key_id, key_secret = settings.require_credentials()
    token, expires_at = _fetch_iam_token(key_id, key_secret)
    _token_cache["token"] = token
    _token_cache["expires_at"] = expires_at
    return token


def get_client() -> OpenAI:
    """Возвращает OpenAI-клиент, авторизованный для настроенного LLM-провайдера.

    Два поддерживаемых типа credentials:
    - `LLM_API_KEY` — статический ключ (cloud.ru Foundation Models API key, ключ
      Ollama/OpenRouter/Groq и т.п.), используется напрямую как Bearer-токен.
    - `LLM_KEY_ID`/`LLM_KEY_SECRET` — ключ сервисного аккаунта cloud.ru, требует
      обмена на короткоживущий IAM access-token (см. `get_access_token`).

    Клиент дешёвый в создании, поэтому пересобираем его при каждом вызове —
    так гарантированно используется свежий токен без ручного отслеживания
    момента протухания на стороне вызывающего кода.
    """
    api_key = settings.llm_api_key or get_access_token()
    return OpenAI(api_key=api_key, base_url=settings.llm_base_url)


def _cache_key(kind: str, payload: dict) -> str:
    raw = json.dumps({"kind": kind, **payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    settings.llm_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings.llm_cache_dir / f"{key}.json"


def _cache_get(key: str) -> dict | None:
    path = _cache_path(key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _cache_set(key: str, payload: dict, result: object) -> None:
    _cache_path(key).write_text(
        json.dumps({"payload": payload, "result": result}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def chat_complete(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    response_format: dict | None = None,
    max_tokens: int | None = None,
    use_cache: bool = True,
) -> str:
    """Один chat-completion запрос, возвращает текст ответа. Кэшируется по содержимому запроса.

    `max_tokens` — обязательная подстраховка для маленьких локальных моделей на
    temperature=0: без верхней границы жадное декодирование иногда срывается в
    повторяющийся луп и генерирует тысячи лишних токенов вместо короткого JSON.
    """

    model = model or settings.llm_chat_model
    payload = {"model": model, "messages": messages, "temperature": temperature}
    if response_format:
        payload["response_format"] = response_format
    if max_tokens:
        payload["max_tokens"] = max_tokens

    key = _cache_key("chat", payload)
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached["result"]

    client = get_client()
    kwargs = dict(model=model, messages=messages, temperature=temperature)
    if response_format:
        kwargs["response_format"] = response_format
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    extra_body = _extra_body()
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""

    if use_cache:
        _cache_set(key, payload, content)
    return content


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _embed_batch(texts: list[str], model: str) -> list[list[float]]:
    client = get_client()
    kwargs = dict(model=model, input=texts)
    extra_body = _extra_body()
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.embeddings.create(**kwargs)
    return [item.embedding for item in response.data]


def embed_texts(
    texts: list[str],
    *,
    model: str | None = None,
    batch_size: int = 20,
    use_cache: bool = True,
) -> np.ndarray:
    """Эмбеддинги для списка текстов. Кэш — по каждому тексту отдельно (хэш текста + модель),
    чтобы добавление новых документов в корпус не требовало пересчёта всего батча."""

    model = model or settings.llm_embed_model
    if not model:
        raise RuntimeError("LLM_EMBED_MODEL не задан в .env")

    results: list[list[float] | None] = [None] * len(texts)
    keys = [_cache_key("embed", {"model": model, "text": t}) for t in texts]

    pending_idx: list[int] = []
    if use_cache:
        for i, key in enumerate(keys):
            cached = _cache_get(key)
            if cached is not None:
                results[i] = cached["result"]
            else:
                pending_idx.append(i)
    else:
        pending_idx = list(range(len(texts)))

    for start in range(0, len(pending_idx), batch_size):
        chunk_idx = pending_idx[start : start + batch_size]
        chunk_texts = [texts[i] for i in chunk_idx]
        embeddings = _embed_batch(chunk_texts, model)
        for i, emb in zip(chunk_idx, embeddings):
            results[i] = emb
            if use_cache:
                _cache_set(keys[i], {"model": model, "text": texts[i]}, emb)

    return np.array(results, dtype=np.float32)


def list_models() -> list[str]:
    client = get_client()
    return sorted(m.id for m in client.models.list().data)


cli = typer.Typer(add_completion=False)


@cli.command("models")
def _cli_models() -> None:
    """Вывести список доступных моделей у настроенного провайдера (для выбора chat/embedding модели в .env).

    Запуск: `uv run python -m graph_rag.llm_client models`.
    """
    for model_id in list_models():
        typer.echo(model_id)


@cli.callback(invoke_without_command=True)
def _cli_main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


if __name__ == "__main__":
    cli()
