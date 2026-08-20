"""Тонкая обёртка над OpenAI-совместимым LLM API (по умолчанию cloud.ru Foundation Models).

Используем официальный `openai` SDK с переопределённым `base_url`, поэтому код
портируем на любой OpenAI-совместимый провайдер (cloud.ru, локальный Ollama,
OpenRouter, Groq и т.п.) — достаточно поменять LLM_BASE_URL/LLM_API_KEY/LLM_*_MODEL
в .env, без правок кода.

Все вызовы кэшируются на диске по хэшу запроса: и эмбеддинги, и chat-completions
дергаются многократно на одном и том же датасете при итеративной разработке пайплайна,
а платный API-вызов — дорогая операция, которую не хочется повторять просто потому,
что перезапустили скрипт.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
import typer
from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from graph_rag.config import settings

_RETRYABLE = (APIConnectionError, RateLimitError, APIError)


class LLMResponseError(RuntimeError):
    """LLM вернул пустой или обрезанный по max_tokens ответ.

    Не входит в `_RETRYABLE` — это не транспортная ошибка, повторный точно такой же
    запрос с тем же max_tokens почти наверняка обрежется точно так же, поэтому решение
    "что делать дальше" (например, повторить с большим max_tokens) оставляем вызывающему
    коду, а не ретраим здесь вслепую. Важно: такой ответ никогда не должен попасть в
    дисковый кэш (см. `chat_complete`) — иначе брак закрепится там навсегда."""


def _extra_body() -> dict | None:
    """Провайдер-специфичные параметры (`LLM_EXTRA_OPTIONS`) как `extra_body` запроса.

    ВНИМАНИЕ, проверено экспериментально: OpenAI-совместимый эндпоинт Ollama
    (`/v1/chat/completions`) молча ИГНОРИРУЕТ поле `options` — запрос с
    `{"options": {"num_predict": 5}}` спокойно генерирует 81 токен. То есть на Ollama
    через этот эндпоинт не применяются ни `num_ctx`, ни `num_thread`, ни `repeat_penalty`.
    Ollama поддерживает только стандартные OpenAI-параметры; её собственные опции
    задаются либо в Modelfile (`ollama create my-model -f Modelfile` с `PARAMETER
    num_ctx 3072`), либо переменными окружения самого сервера Ollama — но не в запросе.

    Поэтому защита от зацикливания маленьких моделей на temperature=0 живёт не здесь, а в
    стандартном `frequency_penalty` (см. `LLM_FREQUENCY_PENALTY` в config) — он
    поддерживается и Ollama, и облачными провайдерами. `LLM_EXTRA_OPTIONS` оставлен для
    провайдеров, которые реально читают `extra_body` (например cloud.ru)."""
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


# Кэш клиента, привязанный к текущему api_key: пересоздаём OpenAI-клиент (а с ним и
# httpx.Client с его пулом TCP/TLS-соединений внутри) только когда api_key реально
# изменился с прошлого раза, а не на каждый вызов. Для статического LLM_API_KEY он
# никогда не меняется — клиент создаётся один раз за весь процесс. Для IAM-токена
# (LLM_KEY_ID/LLM_KEY_SECRET) get_access_token() сам возвращает новый токен ровно тогда,
# когда старый истёк (раз в час, см. _token_cache) — клиент пересобирается синхронно с
# этим, не чаще и не реже, так что долгоживущие батчи не сломаются после ротации токена.
#
# graph_extraction.build_extractions гоняет chat_complete из нескольких потоков
# (ThreadPoolExecutor) — без блокировки первый параллельный вызов на пустом кэше может
# дать несколько потоков, одновременно решивших, что клиента ещё нет, и каждый построит
# свой (сам OpenAI() не делает I/O, так что порчи данных это не вызывает — просто
# лишний неиспользуемый клиент). Лок убирает и это: конструирование клиента сериализуется.
_client_cache: dict[str, object] = {}
_client_cache_lock = threading.Lock()


def get_client() -> OpenAI:
    """Возвращает OpenAI-клиент, авторизованный для настроенного LLM-провайдера.

    Два поддерживаемых типа credentials:
    - `LLM_API_KEY` — статический ключ (cloud.ru Foundation Models API key, ключ
      Ollama/OpenRouter/Groq и т.п.), используется напрямую как Bearer-токен.
    - `LLM_KEY_ID`/`LLM_KEY_SECRET` — ключ сервисного аккаунта cloud.ru, требует
      обмена на короткоживущий IAM access-token (см. `get_access_token`).

    Клиент кэшируется по api_key (см. `_client_cache`) — пересоздаётся только когда
    api_key поменялся, а не на каждый вызов, чтобы переиспользовать TCP/TLS-соединения
    (пул внутри httpx.Client) между запросами, а не открывать новое соединение на
    каждый из сотен LLM-вызовов. Вызывается из нескольких потоков параллельно (см.
    `graph_extraction.build_extractions`), поэтому проверка+запись кэша — под локом.
    """
    api_key = settings.llm_api_key or get_access_token()
    with _client_cache_lock:
        cached_key = _client_cache.get("api_key")
        if cached_key == api_key:
            return _client_cache["client"]  # type: ignore[return-value]

        client = OpenAI(api_key=api_key, base_url=settings.llm_base_url)
        _client_cache["api_key"] = api_key
        _client_cache["client"] = client
        return client


def _cache_key(kind: str, payload: dict) -> str:
    """Ключ дискового кэша — хэш от `payload`, куда для chat-запросов входит полный
    `messages` (включая system prompt). Поэтому правка промпта автоматически даёт
    НОВЫЙ ключ — старые записи под старым ключом никогда не возвращаются по ошибке.

    Но это не бесплатно: старые записи от прошлых версий промпта после такой правки
    никуда не деваются — просто перестают находиться и остаются мусором на диске
    без TTL и автоочистки. Для ручной уборки см. `clear_cache()` / `cache-clear` в CLI
    этого модуля. Для принудительного обхода кэша на конкретном вызове (не то же самое,
    что очистка) см. параметр `use_cache` у `chat_complete`/`embed_texts`."""
    raw = json.dumps({"kind": kind, **payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    settings.llm_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings.llm_cache_dir / f"{key}.json"


def _cache_get(key: str) -> dict | None:
    """Читает запись кэша по ключу. Битый/пустой файл (например, процесс убило посреди
    записи — ноутбук ушёл в сон, kill -9, отключили питание) трактуется как отсутствие
    записи, а не как ошибка: на реальных данных именно так и падал `pipeline build`
    (`JSONDecodeError: Expecting value` на пустом файле кэша) после прерывания долгого
    прогона. `_cache_set` ниже пишет атомарно, так что новых таких файлов появляться не
    должно, но старые, уже лежащие на диске, эта защита тоже переживает."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _cache_set(key: str, payload: dict, result: object) -> None:
    """Пишет запись кэша атомарно: во временный файл в той же директории (та же
    файловая система — гарантия атомарности `os.replace`), затем переименование поверх
    целевого пути. Если процесс убьют посреди записи — останется незавершённый temp-файл,
    а не наполовину записанный (или вовсе пустой) `*.json`, который иначе бы навсегда
    ломал `_cache_get` при следующем запуске (см. её докстринг)."""
    path = _cache_path(key)
    tmp_path = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(
        json.dumps({"payload": payload, "result": result}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def clear_cache() -> int:
    """Удаляет всё содержимое дискового кэша (`settings.llm_cache_dir`) — ручная уборка
    мусора, накопившегося от старых версий промпта (см. `_cache_key`), т.к. TTL/автоочистки
    нет. Возвращает число удалённых файлов. Доступна и как функция, и как CLI-команда
    `cache-clear` (`uv run python -m graph_rag.llm_client cache-clear`)."""
    if not settings.llm_cache_dir.exists():
        return 0
    removed = 0
    for path in settings.llm_cache_dir.glob("*.json"):
        path.unlink()
        removed += 1
    return removed


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
    """Один chat-completion запрос, возвращает текст ответа. Кэшируется по содержимому
    запроса (включая `messages`, то есть и system prompt — см. `_cache_key`), поэтому
    правка промпта сама по себе не потребует чистки кэша руками для НОВЫХ вызовов, но
    старые записи под старым промптом останутся мусором на диске (см. `clear_cache`).

    `use_cache=False` обходит именно этот дисковый кэш на конкретном вызове (не путать
    с его очисткой) — так `build_extractions(force=True)` гарантирует честный новый
    вызов LLM, а не старый закэшированный ответ по тому же промпту.

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
    # Стандартный OpenAI-параметр, работает и на Ollama, и на облачных провайдерах
    # (в отличие от provider-specific опций из `_extra_body`). В payload — значит
    # участвует в ключе кэша: ответы с разным штрафом не должны путаться.
    if settings.llm_frequency_penalty:
        payload["frequency_penalty"] = settings.llm_frequency_penalty

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
    if settings.llm_frequency_penalty:
        kwargs["frequency_penalty"] = settings.llm_frequency_penalty
    extra_body = _extra_body()
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    content = choice.message.content or ""
    finish_reason = choice.finish_reason

    # Проверяем ответ ДО записи в кэш: обрезанный (finish_reason == "length") или пустой
    # контент — не валидный результат, и его нельзя закрепить в дисковом кэше навсегда
    # (--force кэш не чистит, так что брак иначе переживёт даже принудительный перезапуск).
    if finish_reason == "length":
        raise LLMResponseError(
            f"Ответ модели {model!r} обрезан по max_tokens={max_tokens} "
            f"(finish_reason='length'). Увеличь max_tokens или сократи промпт."
        )
    if not content.strip():
        raise LLMResponseError(
            f"Модель {model!r} вернула пустой ответ (finish_reason={finish_reason!r})."
        )

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
    чтобы добавление новых документов в датасет не требовало пересчёта всего батча."""

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


@cli.command("cache-clear")
def _cli_cache_clear() -> None:
    """Удалить весь дисковый кэш LLM-запросов (settings.llm_cache_dir) — например,
    после правки системного промпта, когда старые записи под старым ключом (см.
    `_cache_key`) больше не находятся, но продолжают занимать место на диске.

    Запуск: `uv run python -m graph_rag.llm_client cache-clear`.
    """
    removed = clear_cache()
    typer.echo(f"Удалено файлов кэша: {removed}")


@cli.callback(invoke_without_command=True)
def _cli_main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


if __name__ == "__main__":
    cli()
