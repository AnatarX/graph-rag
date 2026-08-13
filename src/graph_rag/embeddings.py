"""Векторизация корпуса через cloud.ru embeddings API.

Результат — `artifacts/embeddings.npy` (N x D, порядок строк соответствует порядку
документов в `artifacts/docs.parquet`). Сам API-вызов уже кэшируется по хэшу текста
внутри `llm_client.embed_texts`, так что повторный вызов `build_embeddings()` почти
бесплатен — но всё равно сохраняем единый .npy, чтобы клиентскому коду (clustering,
rag) не нужно было тянуть llm_client и знать про кэш.
"""

from __future__ import annotations

import numpy as np

from graph_rag.config import settings
from graph_rag.ingest import load_saved_corpus
from graph_rag.llm_client import embed_texts

EMBEDDINGS_PATH = settings.artifacts_dir / "embeddings.npy"


def build_embeddings(force: bool = False) -> np.ndarray:
    if EMBEDDINGS_PATH.exists() and not force:
        return np.load(EMBEDDINGS_PATH)

    docs = load_saved_corpus()
    vectors = embed_texts(docs["text"].tolist())

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, vectors)
    return vectors


def load_embeddings() -> np.ndarray:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"{EMBEDDINGS_PATH} не найден — сначала прогони build_embeddings().")
    return np.load(EMBEDDINGS_PATH)


if __name__ == "__main__":
    vectors = build_embeddings()
    print(f"Эмбеддинги: {vectors.shape} -> {EMBEDDINGS_PATH}")
