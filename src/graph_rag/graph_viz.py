"""Интерактивная HTML-визуализация графа знаний (pyvis/vis.js) — только для UI.

Отдельный модуль, а не часть graph_store.py: graph_store — чистая работа с данными
(сборка/persist/запросы), без UI-специфичных зависимостей. pyvis тянет за собой JS/HTML
генерацию, которая нужна только Streamlit-приложению.
"""

from __future__ import annotations

import networkx as nx
from pyvis.network import Network

_TYPE_COLORS = {
    "person": "#4C9AFF",
    "organization": "#F5A623",
    "location": "#57D9A3",
    "product": "#D6409F",
    "other": "#9AA5B1",
}


def build_pyvis_html(G: nx.MultiDiGraph, keys: set[str] | None = None, height: str = "600px") -> str:
    """HTML интерактивного графа. Если `keys` задан — только подграф на этих узлах
    (и рёбра между ними), иначе — весь граф. Размер узла ~ степень в подграфе (видно,
    какие сущности — хабы), цвет — тип сущности."""
    nodes = keys if keys is not None else set(G.nodes)
    subgraph = G.subgraph(nodes)
    degree = dict(subgraph.degree())

    net = Network(height=height, width="100%", directed=True, notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-4000, spring_length=140)
    net.set_edge_smooth("dynamic")  # разводит параллельные рёбра между одной парой узлов

    for key in nodes:
        data = G.nodes[key]
        size = min(10 + 3 * degree.get(key, 0), 60)
        net.add_node(
            key,
            label=data["name"],
            title=f"{data['name']} ({data['type']}) — упомянут в {len(data['doc_ids'])} док.",
            color=_TYPE_COLORS.get(data["type"], _TYPE_COLORS["other"]),
            size=size,
        )

    for u, v, data in G.edges(data=True):
        if u in nodes and v in nodes:
            net.add_edge(
                u,
                v,
                label=data["predicate"],
                title=f"{data['predicate']} ({len(data['doc_ids'])} док.: {', '.join(sorted(data['doc_ids'])[:5])})",
                arrows="to",
            )

    return net.generate_html(notebook=False)
