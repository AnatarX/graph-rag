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

# vis.js (внутри pyvis) с "dynamic" edge-smoothing на графе в ~1000+ узлов вешал
# вкладку браузера намертво (проверено вживую). Причина оказалась именно в smooth-режиме
# рёбер, а не в количестве узлов самом по себе — без него граф на 1119 узлов/575 рёбер со
# статичной раскладкой (см. ниже) рендерится нормально. MAX_RENDER_NODES всё равно
# оставлен как защита на уровне функции (не только UI) на случай корпуса заметно больше
# текущего — 100 документов дают не более пары тысяч сущностей, но LLM-извлечение на
# зашумлённых данных иногда плодит дубли/мусорные "сущности" (см. README про entity
# resolution), так что явный потолок дешевле, чем полагаться на размер корпуса.
MAX_RENDER_NODES = 5000

# Минимальный масштаб после auto-fit — ниже этого узлы/подписи превращаются в кашу.
# Намеренно большой: на графе в 1000+ узлов auto-fit почти всегда хочет зумить сильно
# дальше (чтобы влезло всё разом), а нам нужно, чтобы то, что видно на первом экране,
# было крупным и читаемым, даже ценой того, что весь граф целиком не влезает без панорамирования.
_MIN_SCALE = 1.3


def build_pyvis_html(G: nx.MultiDiGraph, keys: set[str] | None = None, height: str = "600px") -> str:
    """HTML интерактивного графа. Если `keys` задан — только подграф на этих узлах
    (и рёбра между ними), иначе — весь граф (с обрезкой до `MAX_RENDER_NODES` самых
    связанных узлов, если исходный граф больше — см. `MAX_RENDER_NODES`). Размер узла ~
    степень в подграфе (видно, какие сущности — хабы), цвет — тип сущности.

    Позиции узлов считаются один раз через `spring_layout` с фиксированным seed и
    отдаются в pyvis с `physics=False` — раскладка детерминирована и не пересчитывается
    силовой симуляцией при каждой перерисовке (иначе на графе в сотни узлов каждый
    ререндер в Streamlit — это заново "трясущаяся" анимация в течение нескольких секунд).
    Узлы всё равно перетаскиваемы мышью — физика влияет только на авто-раскладку.
    Сглаживание рёбер отключено (`smooth: false`) — с "dynamic"/"continuous" на графе
    от нескольких сотен рёбер именно рендер (не layout) вешает вкладку."""
    nodes = keys if keys is not None else set(G.nodes)
    if len(nodes) > MAX_RENDER_NODES:
        degree_full = dict(G.subgraph(nodes).degree())
        nodes = {k for k, _ in sorted(degree_full.items(), key=lambda item: -item[1])[:MAX_RENDER_NODES]}

    subgraph = G.subgraph(nodes)
    degree = dict(subgraph.degree())
    # k — оптимальное расстояние между узлами в spring_layout; дефолт 1/sqrt(n) на графе
    # в 1000+ узлов даёт слишком плотную укладку. Увеличиваем в несколько раз, чтобы
    # между узлами было заметно больше пространства.
    k = 8 / (len(nodes) ** 0.5) if nodes else None
    pos = nx.spring_layout(subgraph, seed=42, k=k) if nodes else {}

    net = Network(height=height, width="100%", directed=True, notebook=False, cdn_resources="in_line")
    net.set_options(
        '{"edges": {"smooth": false}, "physics": {"enabled": false}, '
        # "Весь граф" (1000+ узлов) с всегда видимыми подписями — нечитаемая каша
        # перекрывающегося текста, даже при разумном зуме. drawThreshold прячет подпись,
        # пока сам узел на экране мельче этого числа пикселей — на overview видно только
        # цветные точки (форму графа/кластеры), подписи проявляются по мере приближения.
        # Подсказка (title) при наведении работает независимо от этого, всегда.
        '"nodes": {"scaling": {"label": {"enabled": true, "drawThreshold": 9}}}}'
    )

    for key in nodes:
        data = G.nodes[key]
        size = min(10 + 3 * degree.get(key, 0), 60)
        x, y = pos[key]
        net.add_node(
            key,
            label=data["name"],
            title=f"{data['name']} ({data['type']}) — упомянут в {len(data['doc_ids'])} док.",
            color=_TYPE_COLORS.get(data["type"], _TYPE_COLORS["other"]),
            size=size,
            x=float(x) * 2500,
            y=float(y) * 2500,
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
    # С physics:false vis.js на графах в сотни узлов иногда не делает первый реальный
    # отрисовочный кадр сам (canvas остаётся пустым, хотя network.getPositions() и
    # network.getScale() возвращают нормальные значения — данные загружены, просто не
    # нарисованы) — проверено вживую. network.once("stabilizationIterationsDone", ...),
    # на который обычно полагается vis.js для первого fit+redraw, не срабатывает вовсе,
    # если физика выключена. Форсируем redraw()+fit() явно, а не полагаемся на авто-отрисовку.
    #
    # fit() зумит так, чтобы влезли ВСЕ узлы разом — на графе в 1000+ узлов это зум,
    # при котором подписи и сами узлы становятся нечитаемо мелкими. Клэмпим минимальный
    # масштаб после fit(): если он получился меньше _MIN_SCALE, принудительно
    # приближаем — тогда весь граф целиком может не влезать в первый экран, зато то,
    # что видно, читаемо; остальное — колёсиком мыши/перетаскиванием (граф интерактивный).
    html = html.replace(
        "</body>",
        "<script>setTimeout(function(){ if (typeof network !== 'undefined') { "
        "network.redraw(); network.fit(); "
        f"var s = network.getScale(); var MIN_SCALE = {_MIN_SCALE}; "
        "if (s < MIN_SCALE) { network.moveTo({scale: MIN_SCALE}); } "
        "} }, 50);</script></body>",
    )
    return html
