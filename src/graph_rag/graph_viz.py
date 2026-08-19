"""Интерактивная HTML-визуализация графа знаний (pyvis/vis.js) — только для UI.

Отдельный модуль, а не часть graph_store.py: graph_store — чистая работа с данными
(сборка/persist/запросы), без UI-специфичных зависимостей. pyvis тянет за собой JS/HTML
генерацию, которая нужна только Streamlit-приложению.
"""

from __future__ import annotations

import json

import networkx as nx
from pyvis.network import Network

_TYPE_COLORS = {
    "person": "#4C9AFF",
    "organization": "#F5A623",
    "location": "#57D9A3",
    "product": "#D6409F",
    "other": "#9AA5B1",
}

# vis.js (внутри pyvis) с "dynamic" edge-smoothing на графе в ~1000+ узлов вешал
# вкладку браузера намертво (проверено вживую) — причина в smooth-режиме рёбер, а не в
# физике или количестве узлов самом по себе (см. `_PHYSICS_OPTIONS` — smooth остаётся
# выключенным). MAX_RENDER_NODES всё равно оставлен как защита на уровне функции (не
# только UI) на случай датасета заметно больше текущего — 100 документов дают не более
# пары тысяч сущностей, но LLM-извлечение на зашумлённых данных иногда плодит
# дубли/мусорные "сущности" (см. README про entity resolution), так что явный потолок
# дешевле, чем полагаться на размер датасета.
MAX_RENDER_NODES = 5000

# Минимальный масштаб после auto-fit — ниже этого узлы/подписи превращаются в кашу.
# Намеренно большой: на графе в 1000+ узлов auto-fit почти всегда хочет зумить сильно
# дальше (чтобы влезло всё разом), а нам нужно, чтобы то, что видно на первом экране,
# было крупным и читаемым, даже ценой того, что весь граф целиком не влезает без панорамирования.
_MIN_SCALE = 1.3

# forceAtlas2Based — силовая раскладка vis.js, вариант с почти нулевым centralGravity
# (см. ниже). Первая версия этого модуля использовала barnesHut с дефолтным
# centralGravity=0.3 — эта сила тянет ВСЕ узлы к одной общей точке независимо от связей,
# поэтому на разреженном графе (у нас 573 связей на 983 узла — большинство почти не
# связаны) равновесие "отталкивание против общего притяжения" почти всегда даёт
# равномерный купол/диск, а не органичную форму: изотропное отталкивание вокруг общего
# центра само по себе не может сломать симметрию. centralGravity здесь снижен почти до
# нуля — тогда форму определяют только реальные связи (рёбра-пружины и локальное
# отталкивание), и видны настоящие сгустки/разрежения плотности, а не просто общий
# силуэт. barnesHut с таким низким centralGravity на этом графе оказался численно
# неустойчив (проверено вживую: часть узлов улетает в NaN-позиции — вероятная причина в
# 1/r^2 отталкивании barnesHut, которое расходится при случайном близком старте двух
# узлов) — forceAtlas2Based (линейное, не квадратичное отталкивание) с теми же
# настройками стабилен. Изолированные узлы (degree=0) исключены из полного рендера (см.
# `build_pyvis_html`) по той же физической причине: без единой связи на узел не действует
# ничего, кроме centralGravity, — при centralGravity≈0 у него просто нет силы, которая
# определила бы его позицию осмысленно, и сотни таких узлов лишь размазываются равномерным
# кольцом по периметру, маскируя реальную структуру связанной части графа.
#
# Физику выключаем сразу после стабилизации (см. `_freeze_after_stabilize`) — иначе на
# графе в сотни узлов Streamlit-ререндер каждый раз заново "трясёт" узлы несколько секунд,
# а перетаскивание/зум после заморозки продолжают работать как обычно.
_PHYSICS_OPTIONS = {
    "edges": {"smooth": False},
    "physics": {
        "enabled": True,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
            "gravitationalConstant": -60,
            "centralGravity": 0.006,
            "springLength": 100,
            "springConstant": 0.08,
            "damping": 0.4,
            "avoidOverlap": 0.5,
        },
        "stabilization": {"enabled": True, "iterations": 300, "fit": True},
    },
    # "Весь граф" (1000+ узлов) с всегда видимыми подписями — нечитаемая каша
    # перекрывающегося текста, даже при разумном зуме. drawThreshold прячет подпись,
    # пока сам узел на экране мельче этого числа пикселей — на overview видно только
    # цветные точки (форму графа/кластеры), подписи проявляются по мере приближения.
    # Подсказка (title) при наведении работает независимо от этого, всегда.
    "nodes": {"scaling": {"label": {"enabled": True, "drawThreshold": 9}}},
}


def build_pyvis_html(G: nx.MultiDiGraph, keys: set[str] | None = None, height: str = "600px") -> str:
    """HTML интерактивного графа. Если `keys` задан — только подграф на этих узлах
    (и рёбра между ними). Если не задан — весь граф, но БЕЗ полностью изолированных узлов
    (degree=0, см. обоснование у `_PHYSICS_OPTIONS`) и с обрезкой до `MAX_RENDER_NODES`
    самых связанных узлов, если исходный граф больше (см. `MAX_RENDER_NODES`). Размер
    узла ~ степень в подграфе (видно, какие сущности — хабы), цвет — тип сущности.

    Раскладка — силовая физика vis.js (см. `_PHYSICS_OPTIONS`), не предзаданные позиции:
    узлы стартуют вразброс и расходятся сами за счёт отталкивания + пружин-рёбер, что даёт
    органичный разлёт с видимой неравномерной плотностью вместо статичной раскладки.
    Физика автоматически замораживается, как только раскладка стабилизируется (см.
    `_freeze_after_stabilize`) — дальше узлы можно свободно таскать мышью, но силовая
    симуляция уже не пересчитывается на каждый кадр."""
    nodes = keys if keys is not None else {n for n in G.nodes if G.degree(n) > 0}
    if len(nodes) > MAX_RENDER_NODES:
        degree_full = dict(G.subgraph(nodes).degree())
        nodes = {k for k, _ in sorted(degree_full.items(), key=lambda item: -item[1])[:MAX_RENDER_NODES]}

    subgraph = G.subgraph(nodes)
    degree = dict(subgraph.degree())

    net = Network(height=height, width="100%", directed=True, notebook=False, cdn_resources="in_line")
    net.set_options(json.dumps(_PHYSICS_OPTIONS))

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
                title=f"{data['predicate']} ({len(data['doc_ids'])} док.: {', '.join(sorted(data['doc_ids'])[:5])})",
                arrows="to",
            )

    html = net.generate_html(notebook=False)
    html = html.replace("</body>", _freeze_after_stabilize() + "</body>")
    return html


def _freeze_after_stabilize() -> str:
    """JS: как только vis.js закончит стабилизацию физики — выключить её (замораживает
    позиции) и клэмпнуть минимальный масштаб после auto-fit.

    `stabilizationIterationsDone` — штатное событие vis.js, срабатывает надёжно именно
    потому, что физика включена (в отличие от прежнего physics:false подхода, где это
    событие не срабатывало вовсе и redraw()+fit() приходилось форсировать вручную —
    проверено вживую). `fit()` из `stabilization.fit` в опциях уже подгоняет масштаб под
    все узлы разом — на графе в 1000+ узлов это зум, при котором подписи и сами узлы
    становятся нечитаемо мелкими, поэтому дополнительно клэмпим минимальный масштаб:
    если он получился меньше `_MIN_SCALE`, принудительно приближаем — тогда весь граф
    целиком может не влезать в первый экран, зато то, что видно, читаемо; остальное —
    колёсиком мыши/перетаскиванием (граф интерактивный)."""
    return (
        "<script>if (typeof network !== 'undefined') { "
        "network.once('stabilizationIterationsDone', function() { "
        "network.setOptions({physics: {enabled: false}}); "
        f"var s = network.getScale(); var MIN_SCALE = {_MIN_SCALE}; "
        "if (s < MIN_SCALE) { network.moveTo({scale: MIN_SCALE}); } "
        "}); }</script>"
    )
