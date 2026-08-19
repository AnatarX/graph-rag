"""Тесты на chat_complete: пустой/обрезанный ответ LLM должен кидать LLMResponseError
и никогда не попадать в дисковый кэш. Всё офлайн — OpenAI-клиент подменяется фейком,
никакой реальной сети/ключа не требуется."""

from types import SimpleNamespace

import pytest

from graph_rag import llm_client
from graph_rag.config import settings
from graph_rag.llm_client import LLMResponseError, chat_complete


class FakeCompletions:
    def __init__(self, content: str, finish_reason: str):
        self.content = content
        self.finish_reason = finish_reason
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message, finish_reason=self.finish_reason)
        return SimpleNamespace(choices=[choice])


class FakeClient:
    def __init__(self, content: str, finish_reason: str):
        self.completions = FakeCompletions(content, finish_reason)
        self.chat = SimpleNamespace(completions=self.completions)


def _patch_client(monkeypatch, content: str, finish_reason: str) -> FakeClient:
    fake = FakeClient(content, finish_reason)
    monkeypatch.setattr(llm_client, "get_client", lambda: fake)
    return fake


def _use_tmp_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "llm_cache_dir", tmp_path)


def test_empty_content_raises_llm_response_error_regardless_of_finish_reason(monkeypatch, tmp_path):
    _use_tmp_cache(monkeypatch, tmp_path)
    _patch_client(monkeypatch, content="   ", finish_reason="stop")

    with pytest.raises(LLMResponseError):
        chat_complete([{"role": "user", "content": "hi"}], use_cache=False)


def test_non_empty_content_but_length_finish_reason_raises_llm_response_error(monkeypatch, tmp_path):
    _use_tmp_cache(monkeypatch, tmp_path)
    _patch_client(monkeypatch, content="partial answer that got cut off", finish_reason="length")

    with pytest.raises(LLMResponseError):
        chat_complete([{"role": "user", "content": "hi"}], use_cache=False)


def test_non_empty_content_with_stop_finish_reason_returns_content(monkeypatch, tmp_path):
    _use_tmp_cache(monkeypatch, tmp_path)
    _patch_client(monkeypatch, content="the answer", finish_reason="stop")

    result = chat_complete([{"role": "user", "content": "hi"}], use_cache=False)
    assert result == "the answer"


def test_llm_response_error_is_not_written_to_disk_cache(monkeypatch, tmp_path):
    _use_tmp_cache(monkeypatch, tmp_path)
    _patch_client(monkeypatch, content="", finish_reason="stop")

    with pytest.raises(LLMResponseError):
        chat_complete([{"role": "user", "content": "hi"}], use_cache=True)

    assert list(tmp_path.glob("*.json")) == []


def test_successful_response_is_written_to_disk_cache(monkeypatch, tmp_path):
    _use_tmp_cache(monkeypatch, tmp_path)
    _patch_client(monkeypatch, content="cached answer", finish_reason="stop")

    result = chat_complete([{"role": "user", "content": "hi"}], use_cache=True)
    assert result == "cached answer"
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_cached_result_skips_second_client_call(monkeypatch, tmp_path):
    _use_tmp_cache(monkeypatch, tmp_path)
    fake = _patch_client(monkeypatch, content="cached answer", finish_reason="stop")

    messages = [{"role": "user", "content": "hi"}]
    chat_complete(messages, use_cache=True)
    chat_complete(messages, use_cache=True)

    assert len(fake.completions.calls) == 1
