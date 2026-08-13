"""Минимальный чат-UI поверх rag.answer() — точка входа, не более того."""

import streamlit.components.v1 as components
import streamlit as st

from graph_rag.clustering import CLUSTERS_PATH, load_clusters
from graph_rag.embeddings import EMBEDDINGS_PATH
from graph_rag.graph_store import GRAPH_PATH

st.set_page_config(page_title="Mini GraphRAG · BBC News", page_icon="🕸️", layout="wide")
st.title("🕸️ Mini GraphRAG — BBC News")

if not (EMBEDDINGS_PATH.exists() and GRAPH_PATH.exists()):
    st.warning(
        "Артефакты пайплайна не найдены. Сначала прогони:\n\n"
        "`uv run python -m graph_rag.pipeline build`"
    )
    st.stop()

from graph_rag.graph_store import load_graph, neighborhood_keys  # noqa: E402
from graph_rag.graph_viz import MAX_RENDER_NODES, build_pyvis_html  # noqa: E402
from graph_rag.rag import answer  # noqa: E402 — импорт после проверки артефактов

@st.cache_resource(show_spinner=False)
def _cached_graph():
    return load_graph()


@st.cache_data(show_spinner="Строю визуализацию графа...")
def _cached_graph_html(keys_tuple: tuple[str, ...] | None, height: str) -> str:
    keys = set(keys_tuple) if keys_tuple is not None else None
    return build_pyvis_html(_cached_graph(), keys=keys, height=height)


with st.sidebar:
    st.subheader("Корпус")
    if CLUSTERS_PATH.exists():
        clusters = load_clusters()
        st.caption(f"{clusters['k']} тематических кластеров")
        for cid, c in clusters["clusters"].items():
            st.write(f"**{c['label']}** — {c['size']} док.")

# st.chat_input вызывается на верхнем уровне (не внутри вкладки) — иначе Streamlit не
# прижимает поле ввода ко дну страницы, и оно визуально "уезжает" выше уже отрисованной
# истории чата.
question = st.chat_input("Задай вопрос по корпусу BBC News...")

chat_tab, graph_tab = st.tabs(["💬 Чат", "🕸️ Граф"])

with chat_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Источники"):
                    st.json(message["sources"])

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Ищу контекст и генерирую ответ..."):
                try:
                    result = answer(question)
                except Exception as exc:  # noqa: BLE001 — показываем ошибку пользователю как есть
                    st.error(f"Ошибка при обращении к LLM-провайдеру: {exc}")
                    st.stop()
            st.markdown(result["answer"])
            sources = {
                "matched_entities": result["matched_entities"],
                "documents": result["documents"],
                "graph_facts": result["graph_facts"],
            }
            with st.expander("Источники"):
                st.json(sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "sources": sources}
        )

@st.fragment
def _render_graph_tab() -> None:
    # @st.fragment — виджеты этой вкладки (radio/slider/text_input) перезапускают
    # только эту функцию, а не весь скрипт целиком. Без этого любое взаимодействие
    # с вкладкой "Граф" во время ответа в чате обрывало генерацию ответа: Streamlit
    # при новом прогоне скрипта отменяет предыдущий, ещё выполняющийся — а чат и граф
    # были частью одного и того же линейного скрипта.
    G = _cached_graph()
    st.caption(f"Всего в графе: {G.number_of_nodes()} сущностей, {G.number_of_edges()} связей")

    mode = st.radio(
        "Что показать",
        ["Топ связанных сущностей", "Окрестность сущности", "Весь граф"],
        horizontal=True,
    )

    keys: set[str] | None = None
    if mode == "Топ связанных сущностей":
        n = st.slider("Сколько сущностей", 10, 100, 30)
        degree = dict(G.degree())
        keys = {k for k, _ in sorted(degree.items(), key=lambda item: -item[1])[:n]}
    elif mode == "Окрестность сущности":
        col1, col2 = st.columns([3, 1])
        query = col1.text_input("Сущность (например, Tony Blair)")
        hops = col2.slider("Хопов", 1, 2, 1)
        if query:
            keys = neighborhood_keys(G, query, hops)
            if keys is None:
                st.warning("Не нашёл такую сущность в графе.")
    # mode == "Весь граф" — keys остаётся None, build_pyvis_html рисует весь G
    # (с обрезкой до MAX_RENDER_NODES самых связанных — см. graph_viz.py)

    if mode == "Весь граф" and G.number_of_nodes() > MAX_RENDER_NODES:
        st.caption(
            f"В графе {G.number_of_nodes()} узлов — рендер такого количества виснет в "
            f"браузере (vis.js), показаны {MAX_RENDER_NODES} самых связанных."
        )

    if mode != "Окрестность сущности" or keys:
        # "Весь граф" — заметно больше узлов на экране, чем в остальных режимах, поэтому
        # даём ему больше пикселей: иначе даже с клэмпом минимального зума (graph_viz.py)
        # реально видимая область получается тесной.
        px_height = "1100px" if mode == "Весь граф" else "650px"
        keys_tuple = tuple(sorted(keys)) if keys is not None else None
        html = _cached_graph_html(keys_tuple, px_height)
        components.html(html, height=int(px_height.rstrip("px")) + 20, scrolling=True)


with graph_tab:
    _render_graph_tab()
