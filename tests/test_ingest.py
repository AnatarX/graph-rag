"""Тесты на датасето-независимую часть ingest.py: реестр загрузчиков, нормализацию
схемы (title/category/doc_id) и профиль корпуса.

Всё офлайн: реестр `DATASET_LOADERS` подменяется через monkeypatch синтетическим
загрузчиком, поэтому ни сети (20 Newsgroups), ни реального CSV (BBC) тесты не трогают.
"""

import json

import pandas as pd
import pytest

from graph_rag import ingest
from graph_rag.config import settings
from graph_rag.ingest import (
    TITLE_WORD_COUNT,
    compute_corpus_profile,
    ensure_corpus_profile,
    load_corpus,
    load_corpus_profile,
)


@pytest.fixture
def fake_dataset(monkeypatch):
    """Регистрирует датасет `fake` и делает его текущим; возвращает функцию,
    подменяющую загрузчик."""

    def register(df: pd.DataFrame) -> None:
        monkeypatch.setitem(ingest.DATASET_LOADERS, "fake", lambda: df.copy())
        monkeypatch.setattr(settings, "dataset", "fake")

    return register


def _texts(n: int) -> list[str]:
    return [f"document number {i} about some topic " + "filler word " * 20 for i in range(n)]


def test_title_is_generated_when_loader_has_no_title_column(fake_dataset):
    fake_dataset(pd.DataFrame({"text": _texts(10)}))

    corpus = load_corpus(n_docs=5, seed=0)

    assert "title" in corpus.columns
    for _, row in corpus.iterrows():
        expected = " ".join(row["text"].split()[:TITLE_WORD_COUNT])
        assert row["title"] == expected[0].upper() + expected[1:]


def test_loader_title_column_is_kept_as_is(fake_dataset):
    fake_dataset(pd.DataFrame({"text": _texts(4), "title": [f"Real title {i}" for i in range(4)]}))

    corpus = load_corpus(n_docs=4, seed=0)

    assert sorted(corpus["title"]) == [f"Real title {i}" for i in range(4)]


def test_sampling_works_without_category_column(fake_dataset):
    # Без категорий стратифицировать не по чему — выборка должна быть просто случайной,
    # а не падать на groupby("category").
    fake_dataset(pd.DataFrame({"text": _texts(50)}))

    corpus = load_corpus(n_docs=10, seed=0)

    assert len(corpus) == 10
    assert "category" not in corpus.columns
    assert list(corpus.columns) == ["doc_id", "title", "text"]


def test_sampling_with_category_is_stratified(fake_dataset):
    categories = ["a"] * 40 + ["b"] * 40
    fake_dataset(pd.DataFrame({"text": _texts(80), "category": categories}))

    corpus = load_corpus(n_docs=20, seed=0)

    assert len(corpus) == 20
    assert set(corpus["category"]) == {"a", "b"}
    # Стратификация: доли категорий сохраняются (40/40 -> примерно 10/10).
    assert corpus["category"].value_counts().min() >= 8


def test_n_docs_is_honoured_with_many_categories(fake_dataset):
    # Округление долей по группам недобирает тем сильнее, чем больше категорий
    # (у 20 Newsgroups с 20 категориями получалось 98 документов вместо 100) —
    # недостающее добирается из остатка.
    categories = [f"c{i % 20}" for i in range(600)]
    fake_dataset(pd.DataFrame({"text": _texts(600), "category": categories}))

    corpus = load_corpus(n_docs=100, seed=42)

    assert len(corpus) == 100
    assert corpus["doc_id"].is_unique
    assert corpus["text"].is_unique


def test_doc_id_is_prefixed_with_dataset_name(fake_dataset):
    fake_dataset(pd.DataFrame({"text": _texts(3)}))

    corpus = load_corpus(n_docs=3, seed=0)

    assert list(corpus["doc_id"]) == ["fake-0000", "fake-0001", "fake-0002"]


def test_unknown_dataset_raises_with_available_list(monkeypatch):
    monkeypatch.setattr(settings, "dataset", "definitely-not-a-dataset")

    with pytest.raises(ValueError) as exc_info:
        load_corpus(n_docs=1)

    message = str(exc_info.value)
    assert "definitely-not-a-dataset" in message
    assert "bbc" in message and "20newsgroups" in message


def test_loader_without_text_column_raises(fake_dataset):
    fake_dataset(pd.DataFrame({"body": _texts(3)}))

    with pytest.raises(ValueError, match="text"):
        load_corpus(n_docs=1)


# --- профиль корпуса --------------------------------------------------------


def test_profile_lowercase_corpus_has_no_meaningful_case():
    profile = compute_corpus_profile(
        ["tony blair met gordon brown in london", "microsoft and yukos discussed the law change"]
    )

    assert profile["has_meaningful_case"] is False


def test_profile_normal_case_corpus_has_meaningful_case():
    profile = compute_corpus_profile(
        ["Tony Blair met Gordon Brown in London.", "Microsoft and Yukos discussed the law change."]
    )

    assert profile["has_meaningful_case"] is True


def test_profile_single_stray_uppercase_per_doc_is_not_meaningful_case():
    # Вырожденный случай: заглавная есть в каждом документе, но это артефакт (инициал),
    # а не нормальная капитализация — по доле заглавных среди букв он отсекается.
    profile = compute_corpus_profile(["A " + "lowercase text here " * 50 for _ in range(5)])

    assert profile["docs_with_uppercase_ratio"] == 1.0
    assert profile["has_meaningful_case"] is False


def test_profile_median_doc_chars():
    profile = compute_corpus_profile(["a" * 100, "b" * 300, "c" * 200])

    assert profile["median_doc_chars"] == 200
    assert profile["n_docs"] == 3


def test_load_corpus_profile_falls_back_when_file_missing(tmp_path):
    profile = load_corpus_profile(tmp_path / "corpus_profile.json")

    # Фолбэк = поведение до появления профиля.
    assert profile == {"has_meaningful_case": False, "median_doc_chars": None}


def test_load_corpus_profile_falls_back_on_broken_json(tmp_path):
    path = tmp_path / "corpus_profile.json"
    path.write_text("{ not json", encoding="utf-8")

    assert load_corpus_profile(path)["has_meaningful_case"] is False


def test_ensure_corpus_profile_writes_missing_profile(tmp_path, monkeypatch):
    path = tmp_path / "corpus_profile.json"
    monkeypatch.setattr(ingest, "CORPUS_PROFILE_PATH", path)
    docs = pd.DataFrame({"text": ["Tony Blair met Gordon Brown in London." * 5]})

    profile = ensure_corpus_profile(docs)

    assert path.exists()
    assert profile["has_meaningful_case"] is True
    assert json.loads(path.read_text(encoding="utf-8")) == profile


def test_ensure_corpus_profile_keeps_existing_profile(tmp_path, monkeypatch):
    path = tmp_path / "corpus_profile.json"
    path.write_text(json.dumps({"has_meaningful_case": True, "median_doc_chars": 42}), encoding="utf-8")
    monkeypatch.setattr(ingest, "CORPUS_PROFILE_PATH", path)

    profile = ensure_corpus_profile(pd.DataFrame({"text": ["всё в нижнем регистре"]}))

    assert profile["median_doc_chars"] == 42
