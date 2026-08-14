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
from rapidfuzz import fuzz, process

from graph_rag.embeddings import load_embeddings
from graph_rag.graph_store import GRAPH_PATH, k_hop_neighbors, load_graph
from graph_rag.ingest import load_saved_corpus
from graph_rag.llm_client import chat_complete, embed_texts

TOP_K_DOCS = 5
GRAPH_HOPS = 1
ENTITY_MATCH_SCORE_CUTOFF = 80
MAX_ENTITY_MATCHES = 5
SNIPPET_CHARS = 400

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
    if G.number_of_nodes() == 0:
        return []
    choices = {key: data["name"] for key, data in G.nodes(data=True)}
    matches = process.extract(
        query, choices, scorer=fuzz.partial_ratio, limit=MAX_ENTITY_MATCHES
    )
    return [key for _, score, key in matches if score >= ENTITY_MATCH_SCORE_CUTOFF]


def retrieve(query: str, top_k_docs: int = TOP_K_DOCS, hops: int = GRAPH_HOPS) -> RetrievalResult:
    docs = load_saved_corpus()
    embeddings = load_embeddings()
    query_vec = embed_texts([query])[0]

    doc_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_vec)
    sims = (embeddings @ query_vec) / (doc_norms * query_norm + 1e-8)
    top_idx = np.argsort(-sims)[:top_k_docs]

    vector_hits = [
        VectorHit(
            doc_id=docs.iloc[i]["doc_id"],
            score=float(sims[i]),
            title=docs.iloc[i]["title"],
            snippet=docs.iloc[i]["text"][:SNIPPET_CHARS],
        )
        for i in top_idx
    ]

    graph_facts: list[GraphFact] = []
    matched_entities: list[str] = []
    if GRAPH_PATH.exists():
        G = load_graph()
        for key in _find_query_entities(G, query):
            info = k_hop_neighbors(G, G.nodes[key]["name"], hops=hops)
            if info is None:
                continue
            matched_entities.append(info["entity"])
            for fact in info["facts"]:
                graph_facts.append(GraphFact(**fact))

    return RetrievalResult(vector_hits=vector_hits, graph_facts=graph_facts, matched_entities=matched_entities)


def _build_context_block(retrieval: RetrievalResult) -> str:
    parts = []

    if retrieval.vector_hits:
        parts.append("## Релевантные документы (векторный поиск)")
        for hit in retrieval.vector_hits:
            parts.append(f"[{hit.doc_id}] {hit.title}\n{hit.snippet}")

    if retrieval.graph_facts:
        parts.append("\n## Факты из графа знаний")
        seen = set()
        for fact in retrieval.graph_facts:
            line = f"{fact.subject} --{fact.predicate}--> {fact.object} (источники: {', '.join(fact.doc_ids)})"
            if line not in seen:
                seen.add(line)
                parts.append(line)

    return "\n\n".join(parts)


_CJK_RE = re.compile(r"[一-鿿]")


def _leaked_cjk(text: str, query: str) -> bool:
    """На маленьких локальных моделях (Qwen) изредка "утекает" в китайский —
    известная особенность именно этого семейства моделей, воспроизводится даже на
    temperature=0 детерминированно для конкретных промптов, инструкция в системном
    промпте это не всегда перебивает. Проверяем только когда сам вопрос не на
    китайском — иначе это не утечка, а нормальный ответ на языке вопроса."""
    return bool(_CJK_RE.search(text)) and not _CJK_RE.search(query)


def answer(query: str, top_k_docs: int = TOP_K_DOCS, hops: int = GRAPH_HOPS) -> dict:
    retrieval = retrieve(query, top_k_docs=top_k_docs, hops=hops)
    context = _build_context_block(retrieval)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"},
    ]
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
        "graph_facts": [
            {"subject": f.subject, "predicate": f.predicate, "object": f.object, "doc_ids": f.doc_ids}
            for f in retrieval.graph_facts
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
