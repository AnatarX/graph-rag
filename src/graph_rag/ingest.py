"""Загрузка сырого BBC News CSV и нормализация в единую схему корпуса.

Схема документа: doc_id, category (из датасета — не то же самое, что кластеры из
`clustering.py`, но полезный sanity-check), title (эвристика — первые слова текста,
т.к. в датасете нет отдельного заголовка), text.
"""

from __future__ import annotations

import pandas as pd

from graph_rag.config import settings

TITLE_WORD_COUNT = 12


def _find_raw_csv() -> pd.DataFrame:
    csv_files = sorted(settings.data_raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Нет .csv в {settings.data_raw_dir}. Сначала запусти "
            "`uv run python scripts/download_data.py`."
        )
    return pd.read_csv(csv_files[0])


def _make_title(text: str) -> str:
    words = text.split()
    title = " ".join(words[:TITLE_WORD_COUNT])
    return title[0].upper() + title[1:] if title else ""


def load_corpus(n_docs: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Стратифицированная выборка по категориям, чтобы сохранить структуру корпуса
    в уменьшенном виде, плюс нормализация схемы."""

    n_docs = n_docs or settings.n_docs
    seed = seed if seed is not None else settings.random_seed

    raw = _find_raw_csv()
    raw = raw.dropna(subset=["category", "text"]).reset_index(drop=True)

    frac = min(1.0, n_docs / len(raw))
    sampled = (
        raw.groupby("category", group_keys=False)
        .sample(frac=frac, random_state=seed)
        .sample(frac=1.0, random_state=seed)  # перемешать порядок между категориями
        .reset_index(drop=True)
    )
    sampled = sampled.iloc[:n_docs].reset_index(drop=True)

    sampled["doc_id"] = [f"bbc-{i:04d}" for i in range(len(sampled))]
    sampled["title"] = sampled["text"].map(_make_title)
    sampled = sampled[["doc_id", "category", "title", "text"]]
    return sampled


def save_corpus(df: pd.DataFrame) -> None:
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(settings.artifacts_dir / "docs.parquet", index=False)


def load_saved_corpus() -> pd.DataFrame:
    path = settings.artifacts_dir / "docs.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} не найден — сначала прогони пайплайн ingest.")
    return pd.read_parquet(path)


if __name__ == "__main__":
    corpus = load_corpus()
    save_corpus(corpus)
    print(f"Сохранено {len(corpus)} документов в {settings.artifacts_dir / 'docs.parquet'}")
    print(corpus["category"].value_counts())
