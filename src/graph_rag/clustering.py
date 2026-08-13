"""Кластеризация корпуса по эмбеддингам + человекочитаемые названия кластеров через LLM.

Алгоритм: KMeans поверх L2-нормализованных эмбеддингов (после нормализации евклидово
расстояние KMeans эквивалентно косинусному, которое естественно для эмбеддингов текста).
Число кластеров k не фиксируем — перебираем диапазон и выбираем k с максимальным
silhouette score. Это единственная метрика в задании, что не требует ground truth
(в отличие от, например, сравнения с известными категориями датасета), и работает
устойчиво на малом числе документов.

KMeans выбран вместо HDBSCAN, потому что: (1) корпус маленький и плотный после
нормализации эмбеддингов — кластеры примерно сферические, driving assumption KMeans
разумен; (2) KMeans даёт полное разбиение без "шумовых" точек, что удобнее для
дальнейшего запроса "какие документы принадлежат кластеру N"; (3) не нужно подбирать
чувствительные к масштабу гиперпараметры (min_cluster_size, min_samples), как у
HDBSCAN. HDBSCAN стал бы предпочтительнее на большом и неоднородном по плотности
корпусе, где часть документов не должна принудительно попадать ни в один кластер —
см. README про масштабирование.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from graph_rag.config import settings
from graph_rag.embeddings import load_embeddings
from graph_rag.ingest import load_saved_corpus
from graph_rag.llm_client import chat_complete

CLUSTERS_PATH = settings.artifacts_dir / "clusters.json"
DOC_CLUSTERS_PATH = settings.artifacts_dir / "doc_clusters.parquet"

N_REPRESENTATIVE_DOCS = 5


def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def select_k(
    X: np.ndarray, k_min: int = 2, k_max: int = 15, seed: int = 42
) -> tuple[int, dict[int, float]]:
    """Перебирает k в [k_min, k_max] (обрезая по числу точек) и возвращает k с лучшим
    silhouette score вместе со всеми посчитанными значениями (для прозрачности/логов)."""

    n = len(X)
    k_max = min(k_max, n - 1)
    if k_max < k_min:
        return max(2, min(k_min, n)), {}

    scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
        scores[k] = float(silhouette_score(X, labels))

    best_k = max(scores, key=scores.get)
    return best_k, scores


def _representative_doc_indices(
    X: np.ndarray, labels: np.ndarray, centroids: np.ndarray, cluster_id: int, top_n: int
) -> list[int]:
    member_idx = np.where(labels == cluster_id)[0]
    dists = np.linalg.norm(X[member_idx] - centroids[cluster_id], axis=1)
    order = np.argsort(dists)[:top_n]
    return member_idx[order].tolist()


def label_cluster(representative_titles: list[str]) -> str:
    bullet_list = "\n".join(f"- {t}" for t in representative_titles)
    messages = [
        {
            "role": "system",
            "content": (
                "Ты помогаешь называть тематические кластеры новостных статей. "
                "Ответь ТОЛЬКО коротким названием темы на английском (2-4 слова), "
                "без кавычек и пояснений."
            ),
        },
        {
            "role": "user",
            "content": f"Заголовки представительных документов кластера:\n{bullet_list}",
        },
    ]
    label = chat_complete(messages, temperature=0.0, max_tokens=30)
    return label.strip().strip('"').strip("'")


def build_clusters(force: bool = False) -> dict:
    if CLUSTERS_PATH.exists() and DOC_CLUSTERS_PATH.exists() and not force:
        return json.loads(CLUSTERS_PATH.read_text(encoding="utf-8"))

    docs = load_saved_corpus()
    embeddings = load_embeddings()
    X = l2_normalize(embeddings)

    best_k, scores = select_k(X, seed=settings.random_seed)
    kmeans = KMeans(n_clusters=best_k, random_state=settings.random_seed, n_init=10).fit(X)
    labels = kmeans.labels_

    clusters: dict[str, dict] = {}
    for cluster_id in range(best_k):
        member_idx = np.where(labels == cluster_id)[0]
        rep_idx = _representative_doc_indices(X, labels, kmeans.cluster_centers_, cluster_id, N_REPRESENTATIVE_DOCS)
        rep_titles = docs.iloc[rep_idx]["title"].tolist()
        label = label_cluster(rep_titles)
        clusters[str(cluster_id)] = {
            "label": label,
            "size": int(len(member_idx)),
            "doc_ids": docs.iloc[member_idx]["doc_id"].tolist(),
            "representative_doc_ids": docs.iloc[rep_idx]["doc_id"].tolist(),
        }

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    CLUSTERS_PATH.write_text(
        json.dumps(
            {"k": best_k, "silhouette_scores": scores, "clusters": clusters},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    doc_clusters = pd.DataFrame({"doc_id": docs["doc_id"], "cluster_id": labels.astype(int)})
    doc_clusters.to_parquet(DOC_CLUSTERS_PATH, index=False)

    return json.loads(CLUSTERS_PATH.read_text(encoding="utf-8"))


def load_clusters() -> dict:
    if not CLUSTERS_PATH.exists():
        raise FileNotFoundError(f"{CLUSTERS_PATH} не найден — сначала прогони build_clusters().")
    return json.loads(CLUSTERS_PATH.read_text(encoding="utf-8"))


def load_doc_clusters() -> pd.DataFrame:
    if not DOC_CLUSTERS_PATH.exists():
        raise FileNotFoundError(f"{DOC_CLUSTERS_PATH} не найден — сначала прогони build_clusters().")
    return pd.read_parquet(DOC_CLUSTERS_PATH)


if __name__ == "__main__":
    result = build_clusters()
    print(f"k = {result['k']}")
    for cid, c in result["clusters"].items():
        print(f"[{cid}] {c['label']} ({c['size']} docs)")
