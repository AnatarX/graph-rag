"""Гибридный retrieval (вектор + граф) и генерация ответа через LLM — мини-GraphRAG.

Идея: векторный поиск находит документы, *семантически* похожие на вопрос, но
пропускает связи, которые не отражены в общей теме текста (например, конкретного
человека, упомянутого мельком). Граф добавляет то, что вектор физически не видит:
если в вопросе явно названа сущность, подтягиваем её k-hop окрестность — факты и
документы, где эта сущность встречается, даже если сами документы по общей теме
не похожи на вопрос. Ответ строится на объединении обоих источников контекста, и
каждый источник контекста в ответе помечен так, чтобы было видно, откуда он взялся.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz, process, utils as fuzz_utils

from graph_rag.clustering import l2_normalize
from graph_rag.config import settings
from graph_rag.embeddings import EMBEDDINGS_PATH, load_embeddings
from graph_rag.graph_store import GRAPH_PATH, k_hop_neighbors, load_graph
from graph_rag.ingest import load_saved_corpus
from graph_rag.llm_client import chat_complete, embed_texts

DOCS_PATH = settings.artifacts_dir / "docs.parquet"

# In-process кэш по (путь файла, mtime): retrieve() вызывается на каждый вопрос в чате,
# и без этого docs.parquet/embeddings.npy/graph.json (1157 узлов) перечитывались бы с
# диска заново на каждый вопрос — в UI кэш через @st.cache_resource стоит только на
# вкладке "Граф", чат им не пользуется. Ключ по mtime, а не голый functools.lru_cache():
# в долгоживущем процессе (Streamlit) артефакты пересобираются вручную (`pipeline build`)
# в рамках итеративной разработки, и lru_cache держал бы старые данные в памяти после
# ребилда — это хуже, чем текущее "перечитываем каждый раз". При смене mtime кэш сам
# инвалидируется и подхватывает новый файл.
_load_cache: dict[str, tuple[float, object]] = {}


def _cached(path, loader):
    mtime = path.stat().st_mtime
    cached = _load_cache.get(str(path))
    if cached is not None and cached[0] == mtime:
        return cached[1]
    value = loader()
    _load_cache[str(path)] = (mtime, value)
    return value


TOP_K_DOCS = 5
GRAPH_HOPS = 1
ENTITY_MATCH_SCORE_CUTOFF = 80
MAX_ENTITY_MATCHES = 5
# Медиана длины статьи в корпусе — 2137 символов (максимум 7121), так что этот лимит
# покрывает медианную статью почти целиком, а не только её первые ~19%, как было при 400.
# Полный чанкинг (раздельный эмбеддинг фрагментов документа) — отдельная архитектурная
# задача; здесь прагматичный фикс на уровне окна контекста. Остаток режется по границе
# предложения, а не посреди слова — см. `_truncate_at_sentence`.
SNIPPET_CHARS = 2200
# top_k_docs — потолок, а не гарантия: документы со score ниже порога отсекаются даже
# если top_k ещё не набран (см. `retrieve`). Порог подобран по распределению попарных
# косинусных сходств документов корпуса (l2-нормализованные эмбеддинги, artifacts/embeddings.npy):
# минимум ~0.22, 5-й перцентиль ~0.30, медиана ~0.39 — то есть даже случайная пара статей
# в этом корпусе для этой модели эмбеддингов даёт сходство заметно выше нуля (анизотропия
# эмбеддинг-пространства — общая черта многих sentence-эмбеддеров). 0.2 — с запасом ниже
# этого "фонового шума" корпуса: не отсекает документы, реально похожие на вопрос (пример
# из ревью — 5-й документ со score 0.404 — проходит), но отсекает документы на вопросах,
# которые вообще не по теме корпуса.
MIN_DOC_SCORE = 0.2
# То же число, что и лимит отображения фактов в UI (app/streamlit_app.py, _render_sources:
# `facts[:15]`) — чтобы промпт LLM и то, что видит пользователь, не расходились.
MAX_GRAPH_FACTS_IN_PROMPT = 15
HISTORY_USER_TURNS_FOR_RETRIEVAL = 2
MAX_HISTORY_MESSAGES = 12  # последние 6 пар вопрос/ответ — дальше историю обрезаем

_SYSTEM_PROMPT = """\
Ты отвечаешь на вопросы, используя ТОЛЬКО предоставленный контекст (фрагменты документов
и факты из графа знаний). Если в контексте недостаточно информации — прямо скажи об этом,
не выдумывай. Не копируй фрагменты контекста дословно — перескажи своими словами.

Отвечай на том же языке, на котором задан вопрос (если вопрос по-русски — отвечай
по-русски, даже если исходные документы на английском). Весь ответ целиком должен быть
на этом одном языке — ни слова на каком-либо другом языке, включая китайский.

После утверждений, взятых из источника, указывай его реальный идентификатор в квадратных
скобках, подставляя вместо doc_id настоящее значение из контекста — например, если в
контексте есть фрагмент "[bbc-0035] ...", пиши в ответе [bbc-0035], а не буквально
"[doc_id]"."""


@dataclass
class VectorHit:
    doc_id: str
    score: float
    title: str
    snippet: str


@dataclass
class GraphFact:
    subject: str
    predicate: str
    object: str
    doc_ids: list[str]


@dataclass
class RetrievalResult:
    vector_hits: list[VectorHit] = field(default_factory=list)
    graph_facts: list[GraphFact] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)


def _find_query_entities(G, query: str) -> list[str]:
    """Fuzzy-матчит имена сущностей графа как ЦЕЛЫЕ слова/фразы в вопросе, а не как
    произвольную подстроку: `fuzz.partial_ratio` раньше матчил короткие сущности вроде
    "us" на "b-us-iness" (score 100) — любая сущность короче 4 символов (в графе таких
    десятки) давала такие ложные срабатывания и тащила в промпт чужую окрестность.
    `token_set_ratio` сравнивает по множеству токенов, так что совпадение требует общего
    слова целиком; `processor=default_process` — регистронезависимое сравнение (иначе,
    например, "Tony Blair" в вопросе не матчит узел с отображаемым именем "blair")."""
    if G.number_of_nodes() == 0:
        return []
    choices = {key: data["name"] for key, data in G.nodes(data=True)}
    matches = process.extract(
        query,
        choices,
        scorer=fuzz.token_set_ratio,
        processor=fuzz_utils.default_process,
        limit=MAX_ENTITY_MATCHES,
    )
    return [key for _, score, key in matches if score >= ENTITY_MATCH_SCORE_CUTOFF]


def _contextual_query(query: str, history: list[dict] | None) -> str:
    """Дополняет вопрос именами/темами из недавних реплик пользователя.

    Нужно для follow-up вопросов вида "а когда он родился?" — сам по себе такой
    вопрос не содержит имени, поэтому ни эмбеддинг, ни fuzzy-match сущностей в
    графе (см. `_find_query_entities`) не находят Tony Blair. Берём только
    реплики пользователя (не ответы ассистента) — они короче и обычно как раз
    содержат имя, тогда как ответ ассистента длинный и может размыть эмбеддинг.
    """
    if not history:
        return query
    user_turns = [m["content"] for m in history if m.get("role") == "user"]
    recent = user_turns[-HISTORY_USER_TURNS_FOR_RETRIEVAL:]
    if not recent:
        return query
    return " ".join(recent) + " " + query


def _truncate_at_sentence(text: str, limit: int) -> str:
    """Обрезает текст до `limit` символов по границе предложения, а не голым `text[:limit]`
    посреди слова. Ищет последний знак конца предложения (.!?) в пределах окна; если такого
    нет (например, очень длинное первое предложение), откатывается на границу слова, а если
    и пробела нет — на голый срез (только для патологических случаев без пробелов вообще)."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    last_end = max(window.rfind("."), window.rfind("!"), window.rfind("?"))
    if last_end != -1:
        return window[: last_end + 1]
    last_space = window.rfind(" ")
    if last_space != -1:
        return window[:last_space].rstrip() + "…"
    return window + "…"


def retrieve(query: str, top_k_docs: int = TOP_K_DOCS, hops: int = GRAPH_HOPS) -> RetrievalResult:
    docs = _cached(DOCS_PATH, load_saved_corpus)
    # Нормализуем эмбеддинги документов ОДИН раз на файл, а не на каждый вопрос:
    # `_cached` инвалидируется по mtime, так что пересборка артефактов подхватится, а
    # повторные вопросы в чате переиспользуют уже готовую матрицу. Сама нормализация —
    # l2_normalize из clustering.py, чтобы не дублировать подсчёт норм (см. ревью, п.12).
    doc_unit = _cached(EMBEDDINGS_PATH, lambda: l2_normalize(load_embeddings()))
    query_vec = embed_texts([query])[0]
    query_unit = l2_normalize(query_vec.reshape(1, -1))[0]
    sims = doc_unit @ query_unit

    # top_k_docs — потолок, а не гарантия: сначала отсекаем документы ниже MIN_DOC_SCORE,
    # и только потом берём не более top_k_docs из оставшихся (см. MIN_DOC_SCORE выше).
    ranked_idx = np.argsort(-sims)
    top_idx = [i for i in ranked_idx if sims[i] >= MIN_DOC_SCORE][:top_k_docs]

    vector_hits = [
        VectorHit(
            doc_id=docs.iloc[i]["doc_id"],
            score=float(sims[i]),
            title=docs.iloc[i]["title"],
            snippet=_truncate_at_sentence(docs.iloc[i]["text"], SNIPPET_CHARS),
        )
        for i in top_idx
    ]

    graph_facts: list[GraphFact] = []
    matched_entities: list[str] = []
    if GRAPH_PATH.exists():
        G = _cached(GRAPH_PATH, load_graph)
        for key in _find_query_entities(G, query):
            info = k_hop_neighbors(G, G.nodes[key]["name"], hops=hops)
            if info is None:
                continue
            matched_entities.append(info["entity"])
            for fact in info["facts"]:
                graph_facts.append(GraphFact(**fact))

    return RetrievalResult(vector_hits=vector_hits, graph_facts=graph_facts, matched_entities=matched_entities)


def rank_facts_for_prompt(facts: list[GraphFact]) -> list[GraphFact]:
    """Дедуплицирует факты и оставляет `MAX_GRAPH_FACTS_IN_PROMPT` самых значимых.

    Для хабов (например, "Gordon Brown": 24 факта на 1 хопе, 49 на двух) без ранжирования
    и лимита все факты уезжают в промпт как равнозначные. Значимость меряем числом
    источников (`len(doc_ids)`): подтверждённое несколькими документами важнее единичного
    упоминания. Сортировка стабильна, так что при равном числе источников сохраняется
    порядок обнаружения.

    Публичная (не `_`-приватная) намеренно: `answer()` возвращает ровно этот же список в
    `graph_facts`, чтобы UI показывал пользователю те факты, что реально ушли в промпт
    LLM, а не другой их срез."""
    seen: set[tuple[str, str, str]] = set()
    unique: list[GraphFact] = []
    for fact in facts:
        signature = (fact.subject, fact.predicate, fact.object)
        if signature not in seen:
            seen.add(signature)
            unique.append(fact)
    unique.sort(key=lambda fact: len(fact.doc_ids), reverse=True)
    return unique[:MAX_GRAPH_FACTS_IN_PROMPT]


def _build_context_block(retrieval: RetrievalResult) -> str:
    parts = []

    if retrieval.vector_hits:
        parts.append("## Релевантные документы (векторный поиск)")
        for hit in retrieval.vector_hits:
            parts.append(f"[{hit.doc_id}] {hit.title}\n{hit.snippet}")

    if retrieval.graph_facts:
        parts.append("\n## Факты из графа знаний")
        for fact in rank_facts_for_prompt(retrieval.graph_facts):
            parts.append(
                f"{fact.subject} --{fact.predicate}--> {fact.object} "
                f"(источники: {', '.join(fact.doc_ids)})"
            )

    return "\n\n".join(parts)


_CJK_RE = re.compile(r"[一-鿿]")


def _leaked_cjk(text: str, query: str) -> bool:
    """На маленьких локальных моделях (Qwen) изредка "утекает" в китайский —
    известная особенность именно этого семейства моделей, воспроизводится даже на
    temperature=0 детерминированно для конкретных промптов, инструкция в системном
    промпте это не всегда перебивает. Проверяем только когда сам вопрос не на
    китайском — иначе это не утечка, а нормальный ответ на языке вопроса."""
    return bool(_CJK_RE.search(text)) and not _CJK_RE.search(query)


def answer(
    query: str,
    history: list[dict] | None = None,
    top_k_docs: int = TOP_K_DOCS,
    hops: int = GRAPH_HOPS,
) -> dict:
    """`history` — предыдущие реплики диалога как `[{"role": "user"/"assistant", "content": str}]`,
    без текущего вопроса. Используется и для retrieval (см. `_contextual_query`), и передаётся
    LLM как есть, чтобы follow-up вопросы ("а когда он родился?") разрешались в контексте
    предыдущих реплик, а не терялись."""
    history = history[-MAX_HISTORY_MESSAGES:] if history else history
    search_query = _contextual_query(query, history)
    retrieval = retrieve(search_query, top_k_docs=top_k_docs, hops=hops)
    context = _build_context_block(retrieval)

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"})
    response_text = chat_complete(messages, max_tokens=800, use_cache=False)
    if _leaked_cjk(response_text, query):
        # temperature=0 (жадное декодирование) детерминированно застревает в
        # китайской "колее" для некоторых промптов — небольшая температура иногда
        # даёт модели свернуть на нужный язык. Не гарантия, но одна повторная
        # попытка почти бесплатна по сравнению с уже потраченным временем на первую.
        retry_text = chat_complete(messages, max_tokens=800, use_cache=False, temperature=0.3)
        if not _leaked_cjk(retry_text, query):
            response_text = retry_text

    return {
        "answer": response_text,
        "matched_entities": retrieval.matched_entities,
        "documents": [
            {"doc_id": h.doc_id, "title": h.title, "score": h.score} for h in retrieval.vector_hits
        ],
        # Ровно тот же срез фактов, что ушёл в промпт (см. `rank_facts_for_prompt`):
        # иначе UI показывал бы пользователю одни факты как "источники ответа", а модель
        # отвечала бы по другим.
        "graph_facts": [
            {"subject": f.subject, "predicate": f.predicate, "object": f.object, "doc_ids": f.doc_ids}
            for f in rank_facts_for_prompt(retrieval.graph_facts)
        ],
    }


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is this corpus about?"
    result = answer(question)
    print(result["answer"])
    print("\n--- sources ---")
    for doc in result["documents"]:
        print(f"[{doc['doc_id']}] {doc['title']} (score={doc['score']:.3f})")
