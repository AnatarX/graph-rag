import pandas as pd

from graph_rag.graph_store import (
    attach_clusters,
    build_graph,
    docs_linked_to_cluster,
    k_hop_neighbors,
    shortest_path,
)

EXTRACTIONS = [
    {
        "doc_id": "d1",
        "entities": [
            {"name": "Apple", "type": "organization"},
            {"name": "Tim Cook", "type": "person"},
        ],
        "relations": [{"subject": "Tim Cook", "predicate": "works_for", "object": "Apple"}],
    },
    {
        "doc_id": "d2",
        "entities": [
            {"name": "apple", "type": "organization"},  # разный регистр -> тот же узел
            {"name": "London", "type": "location"},
        ],
        "relations": [{"subject": "Apple", "predicate": "located_in", "object": "London"}],
    },
    {
        "doc_id": "d3",
        "entities": [{"name": "Elon Musk", "type": "person"}],
        "relations": [],
    },
]


def test_case_variants_merge_into_one_node():
    G = build_graph(EXTRACTIONS)
    assert "apple" in G.nodes
    assert G.nodes["apple"]["doc_ids"] == {"d1", "d2"}
    assert {"Apple", "apple"} <= G.nodes["apple"]["aliases"]


def test_shortest_path_across_two_hops():
    G = build_graph(EXTRACTIONS)
    result = shortest_path(G, "Tim Cook", "London")
    assert result is not None
    assert result["path"] == ["Tim Cook", "Apple", "London"] or result["path"] == ["Tim Cook", "apple", "London"]
    assert len(result["steps"]) == 2


def test_shortest_path_none_when_disconnected():
    G = build_graph(EXTRACTIONS)
    assert shortest_path(G, "Elon Musk", "London") is None


def test_k_hop_neighbors():
    G = build_graph(EXTRACTIONS)
    result = k_hop_neighbors(G, "Apple", hops=1)
    assert result["entity"].casefold() == "apple"
    assert "Tim Cook" in result["neighbors"]
    assert "London" in result["neighbors"]
    assert len(result["facts"]) == 2


def test_docs_linked_to_cluster_reaches_beyond_cluster():
    G = build_graph(EXTRACTIONS)
    doc_clusters = pd.DataFrame({"doc_id": ["d1", "d2", "d3"], "cluster_id": [0, 1, 1]})
    attach_clusters(G, doc_clusters)

    result = docs_linked_to_cluster(G, cluster_id=0, doc_clusters=doc_clusters)
    assert result["cluster_doc_ids"] == ["d1"]
    # d1 упоминает Apple, Apple также упомянута в d2 (кластер 1) -> граф "дотягивается" до d2
    assert "d2" in result["extra_doc_ids_via_graph"]
    assert "d3" not in result["linked_doc_ids"]
