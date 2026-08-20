"""Тесты на семантическую (embedding+FAISS+LLM) резолюцию сущностей. Полностью офлайн:
`embed_texts` и `chat_complete` подменяются фейками через monkeypatch (по аналогии с
tests/test_llm_client.py) — реальные вызовы к Ollama/сети здесь не нужны и не делаются."""

from __future__ import annotations

import numpy as np
import pytest
import faiss
import networkx as nx

from graph_rag import entity_resolution as er
from graph_rag.entity_resolution import EntityIndex, judge_candidates, resolve_entity_semantically
from graph_rag.graph_store import build_graph

DIM = 4


def _fake_embed_texts(vectors: dict[str, list[float]]):
    """Фейковый embed_texts: для известного текста возвращает заранее заданный вектор
    (так тест полностью контролирует итоговое косинусное сходство), для неизвестного —
    падает громко, чтобы тест не тихо прошёл на случайных данных. `.calls` — список
    вызовов, чтобы проверять, что embed_texts вообще не звался (или звался нужное число раз)."""

    def fake(texts, **kwargs):
        fake.calls.append(list(texts))
        try:
            rows = [vectors[t] for t in texts]
        except KeyError as exc:
            raise AssertionError(f"unexpected embed_texts call for {exc}") from exc
        return np.array(rows, dtype=np.float32)

    fake.calls = []
    return fake


def _fake_chat_complete(response: str):
    def fake(messages, **kwargs):
        fake.calls.append(messages)
        return response

    fake.calls = []
    return fake


def _un_graph() -> nx.MultiDiGraph:
    """Граф с одним узлом-кандидатом "United Nations", без рёбер — фактов у кандидата
    для этих тестов не требуется, интересна только логика суда над same_as."""
    G = nx.MultiDiGraph()
    G.add_node(
        "united nations",
        name="United Nations",
        type="organization",
        doc_ids={"d1"},
        aliases={"United Nations"},
    )
    return G


def _index_with_un_candidate() -> EntityIndex:
    index = faiss.IndexFlatIP(DIM)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    return EntityIndex(index=index, keys=["united nations"], names=["United Nations"])


# (а) кандидат находится и подтверждается LLM -> резолюция возвращает его ключ
def test_candidate_found_and_confirmed_by_llm_returns_its_key(monkeypatch):
    G = _un_graph()
    entity_index = _index_with_un_candidate()

    fake_embed = _fake_embed_texts({"UN": [0.9, 0.1, 0.0, 0.0]})
    fake_chat = _fake_chat_complete('{"same_as": "united nations"}')
    monkeypatch.setattr(er, "embed_texts", fake_embed)
    monkeypatch.setattr(er, "chat_complete", fake_chat)

    result = resolve_entity_semantically(G, entity_index, "UN", "organization")

    assert result == "united nations"
    assert len(fake_chat.calls) == 1


# (б) кандидат находится, но LLM говорит "не то же самое" -> None
def test_candidate_found_but_llm_says_different_entity_returns_none(monkeypatch):
    G = _un_graph()
    entity_index = _index_with_un_candidate()

    fake_embed = _fake_embed_texts({"UN": [0.9, 0.1, 0.0, 0.0]})
    fake_chat = _fake_chat_complete('{"same_as": null}')
    monkeypatch.setattr(er, "embed_texts", fake_embed)
    monkeypatch.setattr(er, "chat_complete", fake_chat)

    result = resolve_entity_semantically(G, entity_index, "UN", "organization")

    assert result is None
    assert len(fake_chat.calls) == 1


# (в) кандидатов вообще нет (все сходства ниже порога) -> None, LLM даже не вызывается
def test_no_candidates_above_threshold_skips_llm_call(monkeypatch):
    G = _un_graph()
    entity_index = _index_with_un_candidate()

    # Ортогонален кандидату [1,0,0,0] -> косинусное сходство 0, ниже SIMILARITY_THRESHOLD.
    fake_embed = _fake_embed_texts({"Apple": [0.0, 1.0, 0.0, 0.0]})
    fake_chat = _fake_chat_complete('{"same_as": "united nations"}')
    monkeypatch.setattr(er, "embed_texts", fake_embed)
    monkeypatch.setattr(er, "chat_complete", fake_chat)

    result = resolve_entity_semantically(G, entity_index, "Apple", "organization")

    assert result is None
    assert fake_chat.calls == []  # LLM не вызывался вообще


# (г) LLM вернула невалидный same_as (ключ не из списка кандидатов) -> None, не падает
def test_llm_returns_key_outside_candidate_list_is_treated_as_none(monkeypatch):
    fake_chat = _fake_chat_complete('{"same_as": "some-key-that-was-never-offered"}')
    monkeypatch.setattr(er, "chat_complete", fake_chat)

    candidates = [
        {"key": "united nations", "name": "United Nations", "type": "organization", "aliases": [], "facts": []}
    ]
    result = judge_candidates("UN", "organization", candidates)

    assert result is None


@pytest.mark.parametrize(
    "raw_response",
    [
        "not json at all",
        "{}",
        '{"same_as": 42}',
        '{"something_else": "united nations"}',
    ],
)
def test_llm_malformed_or_incomplete_response_is_treated_as_none(monkeypatch, raw_response):
    fake_chat = _fake_chat_complete(raw_response)
    monkeypatch.setattr(er, "chat_complete", fake_chat)

    candidates = [
        {"key": "united nations", "name": "United Nations", "type": "organization", "aliases": [], "facts": []}
    ]
    result = judge_candidates("UN", "organization", candidates)

    assert result is None


# (д) build_graph с resolve_semantically=True и замоканными embed_texts/chat_complete на
# паре синтетических документов -> сущность без общего токена схлопывается в один узел,
# когда мок LLM говорит "да, то же самое". "Russia"/"Russian Federation" — ровно случай из
# ревью: последние токены "russia" и "federation" не совпадают, токенная эвристика тут
# ничего не сделает сама по себе.
def test_build_graph_merges_entity_without_shared_token_via_semantic_resolution(monkeypatch):
    extractions = [
        {"doc_id": "d1", "entities": [{"name": "Russia", "type": "location"}], "relations": []},
        {"doc_id": "d2", "entities": [{"name": "Russian Federation", "type": "location"}], "relations": []},
    ]

    fake_embed = _fake_embed_texts(
        {
            "dimension probe": [1.0, 0.0, 0.0, 0.0],
            "Russia": [1.0, 0.0, 0.0, 0.0],
            "Russian Federation": [0.95, 0.05, 0.0, 0.0],
        }
    )
    fake_chat = _fake_chat_complete('{"same_as": "russia"}')
    monkeypatch.setattr(er, "embed_texts", fake_embed)
    monkeypatch.setattr(er, "chat_complete", fake_chat)

    G = build_graph(extractions, resolve_semantically=True)

    assert G.number_of_nodes() == 1
    node = G.nodes["russia"]
    assert node["doc_ids"] == {"d1", "d2"}
    assert {"Russia", "Russian Federation"} <= node["aliases"]
    assert len(fake_chat.calls) == 1  # ровно один суд, на вторую (новую) сущность


# --- blocking-фильтр перед LLM (has_surface_evidence / _worth_asking_llm) -------------
# Все примеры ниже — реальные пары с прогона по корпусу BBC News: и удачные слияния,
# которые фильтр обязан пропустить, и ложные, которые он обязан остановить до LLM.


@pytest.mark.parametrize(
    "a, b",
    [
        # вложенность токенов
        ("amnesty international", "amnesty"),
        ("circuit city stores", "circuit city"),
        ("barstow, california", "barstow"),
        ("lee bowyer (red card)", "lee bowyer"),
        ("british national party (bnp)", "bnp"),
        # аббревиатура (служебные слова в акроним не входят)
        ("international association of athletics federations", "iaaf"),
        ("european union", "eu"),
        ("court of arbitration for sport", "cas"),
    ],
)
def test_surface_evidence_recognises_name_variants(a, b):
    assert er.has_surface_evidence(a, b)
    assert er.has_surface_evidence(b, a)  # симметрично


@pytest.mark.parametrize(
    "a, b",
    [
        ("cisse", "cage"),
        ("ebell", "ebay"),
        ("bill", "bush"),
        ("anfield", "liverpool"),
        ("mobile gig", "mobile phones"),
        # односложное имя не считается аббревиатурой другого односложного:
        # иначе "3" сошло бы за акроним "3ami" (реальный ложный случай с корпуса)
        ("3", "3ami"),
    ],
)
def test_surface_evidence_rejects_unrelated_names(a, b):
    assert not er.has_surface_evidence(a, b)
    assert not er.has_surface_evidence(b, a)


def test_low_similarity_without_evidence_does_not_reach_llm():
    # 0.559 — реальная похожесть пары "cisse"/"cage", на которой LLM ошибалась
    assert not er._worth_asking_llm("cisse", "cage", 0.559)


def test_high_similarity_reaches_llm_even_without_surface_evidence():
    # "britain"/"uk" — настоящие синонимы без единой общей буквы, 0.837 на реальных данных
    assert er._worth_asking_llm("britain", "uk", 0.837)


def test_low_similarity_with_surface_evidence_reaches_llm():
    # 0.628 — реальная похожесть пары, которую спасает именно вложенность токенов
    assert er._worth_asking_llm("british national party (bnp)", "bnp", 0.628)


def test_find_candidates_blocks_implausible_pair_before_llm(monkeypatch):
    """Кандидат выше порога похожести, но без улик и ниже HIGH_SIMILARITY —
    `find_candidates` не должен его вернуть, то есть LLM не позовут вообще."""
    index = faiss.IndexFlatIP(DIM)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    entity_index = EntityIndex(index=index, keys=["cage"], names=["cage"])

    # ~0.6 похожести: выше SIMILARITY_THRESHOLD (0.55), ниже HIGH_SIMILARITY (0.80)
    fake_embed = _fake_embed_texts({"cisse": [0.6, 0.8, 0.0, 0.0]})
    monkeypatch.setattr(er, "embed_texts", fake_embed)

    assert er.find_candidates(entity_index, "cisse") == []
