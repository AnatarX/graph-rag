"""Граф знаний: сборка networkx-графа из LLM-экстракций, persist, запросы.

Узел — сущность (ключ узла — нормализованное имя: casefold + strip, чтобы схлопывать
"Apple" / "apple" / " Apple " в один узел; отображаемое имя — самый частый вариант
написания). Ребро — направленная связь subject -> object с predicate и списком
doc_id, где эта связь была упомянута.

Точное совпадение после нормализации схлопывает разное написание ОДНОЙ И ТОЙ ЖЕ формы
имени, но не разные формы одного человека в разных документах: LLM извлекает сущность
в том написании, что стоит в конкретном тексте (см. graph_extraction.py), поэтому одна
статья даёт "Tony Blair", а другая — просто "Blair". Без доп. резолюции это были бы два
разных узла графа. `_resolve_node_key` схлопывает такие случаи по правилу "общая
фамилия/последний токен + один набор токенов вложен в другой" (см. `_is_alias_of`) —
токенная эвристика, а не embedding-based entity resolution: осознанное упрощение для
100 документов, дешёвое и достаточно точное на этом масштабе. На большом и шумном
датасете это первое, что перестанет работать (см. README, раздел про масштаб).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd
import typer
from rapidfuzz import fuzz, process

from graph_rag.config import settings
from graph_rag.graph_extraction import is_valid_entity_name

GRAPH_PATH = settings.artifacts_dir / "graph.json"


def _normalize(name: str) -> str:
    return " ".join(name.strip().casefold().split())


def _is_alias_of(a_tokens: list[str], b_tokens: list[str]) -> bool:
    """Считаем `a` другой формой того же имени, что и `b`, если оба заканчиваются
    на один и тот же токен (фамилию/основное имя) и токены короткой формы — подмножество
    токенов длинной. Так "Blair" — алиас "Tony Blair", а "Michael Howard" НЕ схлопывается
    с "Michael Jackson" (общий только первый токен, фамилии разные)."""
    if not a_tokens or not b_tokens or a_tokens[-1] != b_tokens[-1]:
        return False
    shorter, longer = (a_tokens, b_tokens) if len(a_tokens) <= len(b_tokens) else (b_tokens, a_tokens)
    return set(shorter) <= set(longer)


def _resolve_node_key(G: nx.MultiDiGraph, name: str, entity_type: str) -> str:
    """Ключ узла для сущности `name`: существующий узел (точное совпадение или алиас —
    см. `_is_alias_of`) либо новый нормализованный ключ, если совпадений нет.

    Алиасы ищем только среди узлов совместимого типа: точное совпадение типов, либо
    `entity_type == "other"` (сущности, добавленные из relations, где тип неизвестен, —
    там намеренно ищем среди всех типов) или существующий узел ещё имеет тип "other" по
    той же причине."""
    key = _normalize(name)
    if not key or key in G:
        return key
    tokens = key.split()
    for existing_key, data in G.nodes(data=True):
        existing_type = data.get("type", "other")
        if entity_type != "other" and existing_type != "other" and existing_type != entity_type:
            continue
        if _is_alias_of(tokens, existing_key.split()):
            return existing_key
    return key


def build_graph(extractions: list[dict]) -> nx.MultiDiGraph:
    """Собирает граф из LLM-экстракций. Записи, где subject/object/entity — не короткое
    имя сущности, а целая фраза/предложение (LLM это иногда делает, вопреки промпту —
    см. `graph_extraction.is_valid_entity_name`), отбрасываются здесь же: это дешёвая
    защита, которая чистит и уже закэшированные extractions.json задним числом, без
    повторного вызова LLM (сам extract_from_text тоже фильтрует на входе — для новых
    вызовов, — но старые кэшированные данные написаны до этого фильтра)."""
    G = nx.MultiDiGraph()
    name_counts: dict[str, Counter] = {}
    # Тип узла — голосование большинством среди всех НЕ-"other" типов, с которыми
    # сущность встретилась (по всем документам), а не тип первого попавшегося
    # упоминания: LLM не всегда стабильно типизирует одну и ту же сущность одинаково
    # в разных документах, и "первый встреченный" тип — произвольный выбор, который
    # может быть менее частым, чем правильный. Узлы, пришедшие только из relations
    # (там типа вообще нет), сюда ничего не добавляют и остаются "other" — это
    # единственная причина, по которой узел может остаться "other" в итоге.
    type_counts: dict[str, Counter] = {}

    for doc in extractions:
        doc_id = doc["doc_id"]
        for entity in doc["entities"]:
            if not is_valid_entity_name(entity["name"]):
                continue
            etype = entity.get("type", "other")
            key = _resolve_node_key(G, entity["name"], etype)
            if not key:
                continue
            if key not in G:
                G.add_node(key, type="other", doc_ids=set(), aliases=set())
                name_counts[key] = Counter()
            G.nodes[key]["doc_ids"].add(doc_id)
            G.nodes[key]["aliases"].add(entity["name"])
            name_counts[key][entity["name"]] += 1
            if etype != "other":
                type_counts.setdefault(key, Counter())[etype] += 1

        for relation in doc["relations"]:
            if not is_valid_entity_name(relation["subject"]) or not is_valid_entity_name(relation["object"]):
                continue
            skey = _resolve_node_key(G, relation["subject"], "other")
            okey = _resolve_node_key(G, relation["object"], "other")
            predicate = relation["predicate"]
            if not skey or not okey:
                continue
            for key, raw_name in ((skey, relation["subject"]), (okey, relation["object"])):
                if key not in G:
                    G.add_node(key, type="other", doc_ids=set(), aliases=set())
                    name_counts[key] = Counter()
                G.nodes[key]["doc_ids"].add(doc_id)
                G.nodes[key]["aliases"].add(raw_name)
                name_counts[key][raw_name] += 1

            if G.has_edge(skey, okey, key=predicate):
                G.edges[skey, okey, predicate]["doc_ids"].add(doc_id)
            else:
                G.add_edge(skey, okey, key=predicate, predicate=predicate, doc_ids={doc_id})

    for key, counts in name_counts.items():
        G.nodes[key]["name"] = counts.most_common(1)[0][0]
        counts_by_type = type_counts.get(key)
        G.nodes[key]["type"] = counts_by_type.most_common(1)[0][0] if counts_by_type else "other"

    return G


def attach_clusters(G: nx.MultiDiGraph, doc_clusters: pd.DataFrame) -> None:
    """Добавляет каждому узлу множество cluster_id документов, где он упомянут."""
    doc_to_cluster = dict(zip(doc_clusters["doc_id"], doc_clusters["cluster_id"]))
    for _, data in G.nodes(data=True):
        data["cluster_ids"] = sorted({int(doc_to_cluster[d]) for d in data["doc_ids"] if d in doc_to_cluster})


def save_graph(G: nx.MultiDiGraph, path: Path = GRAPH_PATH) -> None:
    nodes = [
        {
            "key": key,
            "name": data["name"],
            "type": data["type"],
            "doc_ids": sorted(data["doc_ids"]),
            "aliases": sorted(data["aliases"]),
            "cluster_ids": data.get("cluster_ids", []),
        }
        for key, data in G.nodes(data=True)
    ]
    edges = [
        {"source": u, "target": v, "predicate": data["predicate"], "doc_ids": sorted(data["doc_ids"])}
        for u, v, data in G.edges(data=True)
    ]
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_graph(path: Path = GRAPH_PATH) -> nx.MultiDiGraph:
    if not path.exists():
        raise FileNotFoundError(f"{path} не найден — сначала собери граф (pipeline build).")
    data = json.loads(path.read_text(encoding="utf-8"))
    G = nx.MultiDiGraph()
    for n in data["nodes"]:
        G.add_node(
            n["key"],
            name=n["name"],
            type=n["type"],
            doc_ids=set(n["doc_ids"]),
            aliases=set(n["aliases"]),
            cluster_ids=n.get("cluster_ids", []),
        )
    for e in data["edges"]:
        G.add_edge(e["source"], e["target"], key=e["predicate"], predicate=e["predicate"], doc_ids=set(e["doc_ids"]))
    return G


# --- запросы по графу -------------------------------------------------------


def resolve_entity(G: nx.MultiDiGraph, query: str, score_cutoff: int = 80) -> str | None:
    """Находит ключ узла по имени: сначала точное совпадение после нормализации,
    иначе fuzzy-match по отображаемым именам (устойчиво к опечаткам/регистру/склонениям).

    `score_cutoff=80` — ниже (было 60) WRatio регулярно выдаёт случайное совпадение с
    коротким/частым именем узла вместо честного "не найдено": на реальном графе
    `resolve_entity("Barack Obama")` (сущности нет в графе) при cutoff=60 находил узел
    вроде "GB"/"Burma" с score в районе 60 — то есть `neighbors "Barack Obama"` тихо
    показывал окрестность совсем другого узла вместо явного сигнала "не найдено"."""
    key = _normalize(query)
    if key in G:
        return key
    choices = {k: d["name"] for k, d in G.nodes(data=True)}
    match = process.extractOne(query, choices, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
    return match[2] if match else None


def shortest_path(G: nx.MultiDiGraph, source: str, target: str) -> dict | None:
    """Путь между двумя сущностями (по неориентированной проекции графа — направление
    связи для вопроса 'через что связаны X и Y' не важно, важна сама цепочка)."""
    skey, tkey = resolve_entity(G, source), resolve_entity(G, target)
    if skey is None or tkey is None:
        return None
    undirected = G.to_undirected(as_view=True)
    try:
        path = nx.shortest_path(undirected, skey, tkey)
    except nx.NetworkXNoPath:
        return None

    steps = []
    for a, b in zip(path, path[1:]):
        # Рёбра могут существовать в оба направления (a->b И b->a, с разными
        # predicate) — собираем предикаты из ОБОИХ направлений, а не только из
        # первого найденного, иначе часть связей между парой узлов теряется молча.
        edge_data = {**(G.get_edge_data(a, b) or {}), **(G.get_edge_data(b, a) or {})}
        predicates = [d["predicate"] for d in edge_data.values()]
        steps.append({"from": G.nodes[a]["name"], "to": G.nodes[b]["name"], "predicates": predicates})
    return {"path": [G.nodes[n]["name"] for n in path], "steps": steps}


def neighborhood_keys(G: nx.MultiDiGraph, entity: str, hops: int = 1) -> set[str] | None:
    """Ключи узлов в k-hop окрестности сущности (включая саму сущность), или None,
    если сущность не найдена. Используется и для фактов (`k_hop_neighbors`), и для
    подграфа при визуализации (`graph_viz.build_pyvis_html`)."""
    key = resolve_entity(G, entity)
    if key is None:
        return None
    undirected = G.to_undirected(as_view=True)
    return set(nx.single_source_shortest_path_length(undirected, key, cutoff=hops))


def k_hop_neighbors(G: nx.MultiDiGraph, entity: str, hops: int = 1) -> dict | None:
    """Всё, что связано с сущностью через 1-2 хопа: соседи + факты (триплеты) между
    узлами внутри этой окрестности."""
    lengths = neighborhood_keys(G, entity, hops)
    if lengths is None:
        return None
    key = resolve_entity(G, entity)

    facts = []
    for a, b, data in G.edges(data=True):
        if a in lengths and b in lengths:
            facts.append(
                {
                    "subject": G.nodes[a]["name"],
                    "predicate": data["predicate"],
                    "object": G.nodes[b]["name"],
                    "doc_ids": sorted(data["doc_ids"]),
                }
            )

    return {
        "entity": G.nodes[key]["name"],
        "neighbors": sorted(G.nodes[n]["name"] for n in lengths if n != key),
        "facts": facts,
    }


def docs_linked_to_cluster(G: nx.MultiDiGraph, cluster_id: int, doc_clusters: pd.DataFrame) -> dict:
    """Документы кластера N + документы ВНЕ кластера, до которых можно дойти через
    общие сущности — это ровно то место, где граф даёт ценность поверх векторной
    кластеризации: документы, не похожие по эмбеддингу целиком, но связанные через
    конкретную сущность (человека/компанию/место).

    "Мостовой" считается сущность, встреченная хотя бы в одном документе кластера —
    но документы ВНЕ кластера ранжируются по числу РАЗНЫХ мостовых сущностей, которые
    их связывают с кластером (а не просто перечисляются все как одна плоская "половина
    корпуса"): документ, разделяющий с кластером 5 сущностей, — куда более осмысленная
    находка, чем документ с одной случайной общей сущностью, и вызывающий код должен это
    видеть."""
    cluster_doc_ids = set(doc_clusters.loc[doc_clusters["cluster_id"] == cluster_id, "doc_id"])
    bridge_entities = [
        (key, data["name"]) for key, data in G.nodes(data=True) if data["doc_ids"] & cluster_doc_ids
    ]
    bridge_keys = [key for key, _ in bridge_entities]

    linked_doc_ids: set[str] = set()
    shared_counts: Counter[str] = Counter()
    for key in bridge_keys:
        doc_ids = G.nodes[key]["doc_ids"]
        linked_doc_ids |= doc_ids
        for doc_id in doc_ids - cluster_doc_ids:
            shared_counts[doc_id] += 1

    extra_doc_ids_via_graph = [
        {"doc_id": doc_id, "shared_entities": count}
        for doc_id, count in sorted(shared_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    return {
        "cluster_doc_ids": sorted(cluster_doc_ids),
        "bridge_entities": sorted(name for _, name in bridge_entities),
        "linked_doc_ids": sorted(linked_doc_ids),
        "extra_doc_ids_via_graph": extra_doc_ids_via_graph,
    }


# --- CLI: чтобы три запроса выше можно было потрогать руками, а не только из кода ----

cli = typer.Typer(add_completion=False, help="Запросы по графу знаний.")


@cli.command("path")
def _cli_path(source: str, target: str) -> None:
    """Через какие сущности связаны X и Y."""
    result = shortest_path(load_graph(), source, target)
    if result is None:
        typer.echo("Путь не найден (или одна из сущностей отсутствует в графе).")
        raise typer.Exit(1)
    typer.echo(" -> ".join(result["path"]))
    for step in result["steps"]:
        typer.echo(f"  {step['from']} --{'/'.join(step['predicates'])}--> {step['to']}")


@cli.command("neighbors")
def _cli_neighbors(entity: str, hops: int = typer.Option(1, "--hops")) -> None:
    """Всё, что связано с сущностью через 1-2 хопа."""
    result = k_hop_neighbors(load_graph(), entity, hops=hops)
    if result is None:
        typer.echo(f"Сущность {entity!r} не найдена в графе.")
        raise typer.Exit(1)
    typer.echo(f"{result['entity']}: {len(result['neighbors'])} соседей, {len(result['facts'])} фактов")
    for fact in result["facts"]:
        typer.echo(f"  {fact['subject']} --{fact['predicate']}--> {fact['object']}  {fact['doc_ids']}")


@cli.command("cluster")
def _cli_cluster(cluster_id: int) -> None:
    """Документы кластера N + документы, до которых граф дотягивается через общие сущности."""
    from graph_rag.clustering import load_doc_clusters

    result = docs_linked_to_cluster(load_graph(), cluster_id, load_doc_clusters())
    typer.echo(f"документов в кластере:        {len(result['cluster_doc_ids'])}")
    typer.echo(f"связанных через граф (всего): {len(result['linked_doc_ids'])}")
    typer.echo(f"из них ВНЕ кластера:          {len(result['extra_doc_ids_via_graph'])}")
    typer.echo("  ранжированы по числу общих (мостовых) сущностей с кластером:")
    for entry in result["extra_doc_ids_via_graph"]:
        typer.echo(f"    {entry['doc_id']}  (общих сущностей: {entry['shared_entities']})")


if __name__ == "__main__":
    cli()
