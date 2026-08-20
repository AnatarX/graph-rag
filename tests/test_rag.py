"""Тесты на офлайн-часть rag.py: fuzzy-матчинг сущностей запроса по графу
(`_find_query_entities`), сборку контекстного блока для промпта (`_build_context_block`),
обрезку сниппетов по границе предложения (`_truncate_at_sentence`) и выбор длины сниппета
по профилю корпуса (`snippet_chars`). Ничего из этого не трогает сеть/LLM API —
retrieve()/answer() здесь не вызываются."""

import networkx as nx

from graph_rag.rag import (
    MAX_GRAPH_FACTS_IN_PROMPT,
    MAX_SNIPPET_CHARS,
    MIN_SNIPPET_CHARS,
    SNIPPET_CHARS,
    GraphFact,
    RetrievalResult,
    VectorHit,
    _build_context_block,
    _find_query_entities,
    _truncate_at_sentence,
    snippet_chars,
)


def test_snippet_chars_uses_corpus_median():
    assert snippet_chars({"median_doc_chars": 1500}) == 1500


def test_snippet_chars_clamps_extremes():
    # Слишком короткий сниппет режет документ до огрызка, слишком длинный — раздувает промпт.
    assert snippet_chars({"median_doc_chars": 50}) == MIN_SNIPPET_CHARS
    assert snippet_chars({"median_doc_chars": 100_000}) == MAX_SNIPPET_CHARS


def test_snippet_chars_falls_back_without_profile():
    # Профиля нет (старые артефакты) — работаем по прежней константе.
    assert snippet_chars({"median_doc_chars": None}) == SNIPPET_CHARS
    assert snippet_chars({}) == SNIPPET_CHARS


def _build_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    G.add_node("us", name="US")
    G.add_node("uk", name="UK")
    G.add_node("gordon brown", name="Gordon Brown")
    G.add_node("apple", name="Apple")
    return G


def test_find_query_entities_does_not_match_short_node_as_substring():
    # Regression из ревью: раньше fuzz.partial_ratio матчил короткую сущность "us"
    # как подстроку "b-us-iness" в "business" (score=100). token_set_ratio с
    # default_process требует общего токена целиком, поэтому "us" не должен
    # попадать в результат на вопросе, где нет слова "us" как отдельного токена.
    G = _build_graph()
    matches = _find_query_entities(G, "what business trends are discussed?")
    assert "us" not in matches


def test_find_query_entities_matches_explicit_full_entity_name():
    G = _build_graph()
    matches = _find_query_entities(G, "Gordon Brown gave a speech about the economy today")
    assert "gordon brown" in matches


def test_find_query_entities_empty_graph_returns_empty_list():
    G = nx.MultiDiGraph()
    assert _find_query_entities(G, "anything at all") == []


def test_truncate_at_sentence_leaves_short_text_untouched():
    text = "hello world"
    assert _truncate_at_sentence(text, 50) == text


def test_truncate_at_sentence_without_punctuation_cuts_at_word_boundary():
    text = "word " * 20
    result = _truncate_at_sentence(text, 40)
    assert result == "word word word word word word word word…"
    assert result.endswith("…")


def test_truncate_at_sentence_cuts_at_last_period_within_limit():
    text = "First sentence here. Second sentence goes further and further. Third trails off"
    result = _truncate_at_sentence(text, 40)
    assert result == "First sentence here."


def _fact(subject: str, predicate: str, obj: str, doc_ids: list[str]) -> GraphFact:
    return GraphFact(subject=subject, predicate=predicate, object=obj, doc_ids=doc_ids)


def test_build_context_block_sorts_facts_by_number_of_sources_descending():
    low = _fact("A", "p1", "B", ["d1"])
    high = _fact("C", "p2", "D", ["d1", "d2", "d3"])
    mid = _fact("E", "p3", "F", ["d1", "d2"])

    retrieval = RetrievalResult(graph_facts=[low, high, mid])
    block = _build_context_block(retrieval)

    pos_high = block.index("C --p2--> D")
    pos_mid = block.index("E --p3--> F")
    pos_low = block.index("A --p1--> B")
    assert pos_high < pos_mid < pos_low


def test_build_context_block_truncates_to_max_graph_facts():
    facts = [
        _fact(f"S{i}", "rel", f"O{i}", [f"d{i}"])
        for i in range(MAX_GRAPH_FACTS_IN_PROMPT + 5)
    ]
    retrieval = RetrievalResult(graph_facts=facts)
    block = _build_context_block(retrieval)
    assert block.count("--rel-->") == MAX_GRAPH_FACTS_IN_PROMPT


def test_build_context_block_deduplicates_identical_facts():
    fact_a = _fact("A", "p1", "B", ["d1", "d2"])
    fact_a_dup = _fact("A", "p1", "B", ["d1", "d2"])
    fact_b = _fact("C", "p2", "D", ["d1"])

    retrieval = RetrievalResult(graph_facts=[fact_a, fact_a_dup, fact_b])
    block = _build_context_block(retrieval)
    assert block.count("A --p1--> B") == 1
    assert block.count("C --p2--> D") == 1


def test_build_context_block_includes_vector_hits_section():
    hit = VectorHit(doc_id="bbc-0001", score=0.5, title="Some title", snippet="Some snippet.")
    retrieval = RetrievalResult(vector_hits=[hit])
    block = _build_context_block(retrieval)
    assert "[bbc-0001] Some title" in block
    assert "Some snippet." in block


def test_build_context_block_empty_retrieval_returns_empty_string():
    assert _build_context_block(RetrievalResult()) == ""
