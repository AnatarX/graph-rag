import numpy as np
from sklearn.datasets import make_blobs

from graph_rag.clustering import l2_normalize, select_k


def test_select_k_finds_well_separated_clusters():
    X, _ = make_blobs(
        n_samples=120,
        centers=4,
        n_features=8,
        cluster_std=0.5,
        random_state=0,
    )
    best_k, scores = select_k(X, k_min=2, k_max=8, seed=0)
    assert best_k == 4
    assert scores[4] == max(scores.values())


def test_l2_normalize_unit_norm():
    X = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])
    normalized = l2_normalize(X)
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms[:2], 1.0)
    assert np.allclose(normalized[2], 0.0)  # нулевой вектор остаётся нулевым, не NaN
