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
    # resolve_semantically=False во всех тестах этого файла: они проверяют именно
    # токенную эвристику (_resolve_node_key/_is_alias_of) офлайн и детерминированно, без
    # сети/LLM/эмбеддингов. Семантический (embedding+FAISS+LLM) слой резолюции тестируется
    # отдельно, полностью замоканным, в tests/test_entity_resolution.py.
    G = build_graph(EXTRACTIONS, resolve_semantically=False)
    assert "apple" in G.nodes
    assert G.nodes["apple"]["doc_ids"] == {"d1", "d2"}
    assert {"Apple", "apple"} <= G.nodes["apple"]["aliases"]


def test_same_person_different_name_forms_merge_into_one_node():
    extractions = [
        {
            "doc_id": "d1",
            "entities": [{"name": "Tony Blair", "type": "person"}],
            "relations": [],
        },
        {
            "doc_id": "d2",
            # Другая статья называет его просто по фамилии — должно схлопнуться
            # в тот же узел, что и "Tony Blair", а не создать дубль.
            "entities": [{"name": "Blair", "type": "person"}],
            "relations": [],
        },
    ]
    G = build_graph(extractions, resolve_semantically=False)
    assert G.number_of_nodes() == 1
    node = G.nodes["tony blair"]
    assert node["doc_ids"] == {"d1", "d2"}
    assert {"Tony Blair", "Blair"} <= node["aliases"]


def test_shared_first_name_does_not_merge_different_people():
    extractions = [
        {
            "doc_id": "d1",
            "entities": [{"name": "Michael Howard", "type": "person"}],
            "relations": [],
        },
        {
            "doc_id": "d2",
            "entities": [{"name": "Michael Jackson", "type": "person"}],
            "relations": [],
        },
    ]
    G = build_graph(extractions, resolve_semantically=False)
    assert G.number_of_nodes() == 2


def test_sentence_like_entity_and_relation_participants_are_dropped():
    extractions = [
        {
            "doc_id": "d1",
            "entities": [
                {"name": "Tony Blair", "type": "person"},
                # LLM иногда подставляет целую фразу вместо короткого имени сущности —
                # см. graph_extraction.is_valid_entity_name.
                {"name": "he probably has his own airplane seat that is how highly sony prize him", "type": "other"},
            ],
            "relations": [
                {"subject": "Tony Blair", "predicate": "leads", "object": "UK"},
                {
                    "subject": "Tony Blair",
                    "predicate": "caused",
                    "object": "52% rise in profits for the year to £198m from the £130m seen a year earlier",
                },
            ],
        }
    ]
    G = build_graph(extractions, resolve_semantically=False)
    assert "tony blair" in G.nodes
    assert "uk" in G.nodes
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1


def test_shortest_path_across_two_hops():
    G = build_graph(EXTRACTIONS, resolve_semantically=False)
    result = shortest_path(G, "Tim Cook", "London")
    assert result is not None
    assert result["path"] == ["Tim Cook", "Apple", "London"] or result["path"] == ["Tim Cook", "apple", "London"]
    assert len(result["steps"]) == 2


def test_shortest_path_none_when_disconnected():
    G = build_graph(EXTRACTIONS, resolve_semantically=False)
    assert shortest_path(G, "Elon Musk", "London") is None


def test_k_hop_neighbors():
    G = build_graph(EXTRACTIONS, resolve_semantically=False)
    result = k_hop_neighbors(G, "Apple", hops=1)
    assert result["entity"].casefold() == "apple"
    assert "Tim Cook" in result["neighbors"]
    assert "London" in result["neighbors"]
    assert len(result["facts"]) == 2


def test_docs_linked_to_cluster_reaches_beyond_cluster():
    G = build_graph(EXTRACTIONS, resolve_semantically=False)
    doc_clusters = pd.DataFrame({"doc_id": ["d1", "d2", "d3"], "cluster_id": [0, 1, 1]})
    attach_clusters(G, doc_clusters)

    result = docs_linked_to_cluster(G, cluster_id=0, doc_clusters=doc_clusters)
    assert result["cluster_doc_ids"] == ["d1"]
    # d1 упоминает Apple, Apple также упомянута в d2 (кластер 1) -> граф "дотягивается" до d2,
    # с указанием, через сколько общих сущностей (тут Apple и Tim Cook -> и Apple связывает
    # d1/d2, при этом Tim Cook в d2 не упомянут, так что общих сущностей ровно 1: Apple).
    extra_doc_ids = [entry["doc_id"] for entry in result["extra_doc_ids_via_graph"]]
    assert "d2" in extra_doc_ids
    assert all("shared_entities" in entry for entry in result["extra_doc_ids_via_graph"])
    assert "d3" not in result["linked_doc_ids"]
