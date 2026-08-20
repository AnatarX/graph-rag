"""Семантическая (embedding + FAISS + LLM) резолюция сущностей — второй проход резолюции,
дополняющий токенную эвристику в `graph_store._resolve_node_key`.

Токенная эвристика (общая фамилия/последний токен + вложенность токенов) бесплатна и хорошо
ловит случаи вроде "Blair" / "Tony Blair", но ничего не может с синонимами без общего токена:
"US" / "United States" / "America", "UN" / "United Nations", "Russia" / "Russian Federation".
Именно такие случаи и должен закрывать этот модуль.

Как это устроено (см. README, раздел "Ключевые решения", для более подробного описания):

1. У каждой новой сущности эмбеддится ИМЯ (не весь контекст) через `embed_texts` (bge-m3).
2. Вектор ищется в инкрементальном FAISS-индексе (`IndexFlatIP` поверх L2-нормализованных
   векторов — при такой нормализации inner product эквивалентен косинусному сходству), куда
   заранее сложены имена уже существующих узлов графа. Возвращаются top-k соседей выше
   `SIMILARITY_THRESHOLD`.
3. Порог сознательно рыхлый: эмпирическая калибровка на паре десятков реальных имён сущностей
   (см. README) показала, что настоящие синонимы дают cosine ~0.85-0.88 ("Russia"/"Russian
   Federation" = 0.851, "United Nations"/"UN" = 0.879), но и совершенно разные сущности иногда
   дают 0.6-0.65 ("Apple"/"Microsoft" = 0.611, "Gordon Brown"/"Michael Jackson" = 0.625) —
   шумового провала между "точно да" и "точно нет" нет. Поэтому FAISS-шаг оптимизирован на
   recall (не пропустить кандидата), а не на precision: `SIMILARITY_THRESHOLD = 0.55` ниже
   всех наблюдавшихся синонимов, но выше почти всего явного шума.
4. Финальное решение "это та же сущность или нет" — за LLM: ей показывают имя+тип новой
   сущности и каждого кандидата, обогащённого алиасами и несколькими фактами (рёбрами), в
   которых кандидат участвует как subject или object. Промпт явно предупреждает про шумовой
   потолок похожести и просит НЕ мержить при малейшей неуверенности: ложный мерж (два разных
   человека/организации в одном узле) портит граф сильнее, чем недомерж (граф просто остаётся
   с двумя узлами вместо одного — status quo, с которым система и так живёт).

Модуль только *резолвит* — решает, что новая сущность совпадает (или нет) с существующим узлом.
Он не мутирует граф и не решает, когда добавлять эмбеддинг в индекс — это ответственность
вызывающего кода (`graph_store.build_graph`), у которого для этого есть весь граф целиком.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import faiss
import networkx as nx
import numpy as np

from graph_rag.clustering import l2_normalize
from graph_rag.llm_client import chat_complete, embed_texts

# Ниже почти всех наблюдавшихся синонимов (0.85+) и почти всего наблюдавшегося шума (0.6-0.65)
# по замерам из README — но рыхлый специально: задача этого порога — recall (не отсечь
# кандидата раньше времени), а не итоговое решение, которое всё равно за LLM.
SIMILARITY_THRESHOLD = 0.55
TOP_K_CANDIDATES = 5

# Кандидату хватает пары фактов, чтобы LLM могла понять контекст, не раздувая промпт.
_MAX_FACTS_PER_CANDIDATE = 4
_JUDGE_MAX_TOKENS = 80


@dataclass
class EntityIndex:
    """Инкрементальный FAISS-индекс имён узлов графа: `index` хранит только векторы
    (L2-нормализованные, cosine similarity через inner product), `keys[i]` — ключ узла
    графа, отвечающий вектору на позиции `i` (FAISS сам метаданные не хранит)."""

    index: faiss.IndexFlatIP
    keys: list[str] = field(default_factory=list)


def create_index() -> EntityIndex:
    """Пустой FAISS-индекс нужной размерности. Размерность берётся реальным вызовом
    `embed_texts`, а не хардкодится — она зависит от настроенной embedding-модели
    (для bge-m3 сейчас 1024, но это не должно быть magic-числом в коде)."""
    dim = embed_texts(["dimension probe"])[0].shape[0]
    return EntityIndex(index=faiss.IndexFlatIP(int(dim)))


def add_to_index(entity_index: EntityIndex, key: str, name: str) -> None:
    """Эмбеддит `name` и добавляет вектор в индекс, привязывая его к ключу узла `key`."""
    vector = l2_normalize(embed_texts([name]).astype(np.float32))
    entity_index.index.add(vector)
    entity_index.keys.append(key)


def find_candidates(
    entity_index: EntityIndex,
    name: str,
    *,
    top_k: int = TOP_K_CANDIDATES,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[str]:
    """Ключи узлов-кандидатов: топ-k ближайших соседей `name` в индексе по косинусному
    сходству, отфильтрованные по `threshold`. Никаких обращений к LLM здесь — чисто
    векторный поиск, LLM-суд — отдельный шаг (`judge_candidate`/`resolve_entity_semantically`)."""
    if entity_index.index.ntotal == 0:
        return []
    vector = l2_normalize(embed_texts([name]).astype(np.float32))
    k = min(top_k, entity_index.index.ntotal)
    scores, indices = entity_index.index.search(vector, k)
    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < threshold:
            continue
        candidates.append(entity_index.keys[idx])
    return candidates


def _candidate_facts(G: nx.MultiDiGraph, key: str, limit: int = _MAX_FACTS_PER_CANDIDATE) -> list[str]:
    """До `limit` фактов (триплетов), где `key` участвует как subject или object —
    контекст для LLM-суда, откуда видно, о чём вообще идёт речь про кандидата, а не только
    его имя. Собирает из обоих направлений рёбер, обрезая, а не пытаясь быть исчерпывающим."""
    facts: list[str] = []
    for _, target, data in G.out_edges(key, data=True):
        facts.append(f"{G.nodes[key]['name']} --{data['predicate']}--> {G.nodes[target]['name']}")
        if len(facts) >= limit:
            return facts
    for source, _, data in G.in_edges(key, data=True):
        facts.append(f"{G.nodes[source]['name']} --{data['predicate']}--> {G.nodes[key]['name']}")
        if len(facts) >= limit:
            return facts
    return facts


def _describe_candidate(G: nx.MultiDiGraph, key: str) -> dict:
    data = G.nodes[key]
    return {
        "key": key,
        "name": data["name"],
        "type": data.get("type", "other"),
        "aliases": sorted(data.get("aliases", set())),
        "facts": _candidate_facts(G, key),
    }


_JUDGE_SYSTEM_PROMPT = """Ты помогаешь строить граф знаний и решаешь один узкий вопрос:
является ли НОВАЯ сущность тем же реальным объектом, что и один из сущностей-КАНДИДАТОВ,
уже присутствующих в графе.

Кандидаты подобраны по эмбеддинг-похожести имён, но это ненадёжный сигнал: похожесть 0.6-0.65
регулярно встречается у СОВЕРШЕННО РАЗНЫХ сущностей (например "Apple"/"Microsoft" или
"Gordon Brown"/"Michael Jackson"), а не только у настоящих синонимов. Поэтому не доверяй
самому факту, что кандидат был предложен, — рассуждай по существу: одинаковый тип, одинаковые
или пересекающиеся алиасы, согласующиеся факты.

Если сомневаешься — отвечай "нет". Ложное слияние (объединить в один узел два разных реальных
объекта) портит граф куда сильнее, чем отказ от слияния (граф просто останется с двумя
отдельными узлами вместо одного).

Ответь СТРОГО JSON без пояснений: {"same_as": "<ключ кандидата>"} если один из кандидатов —
та же сущность, или {"same_as": null} если ни один не подходит (или ты не уверена)."""


def judge_candidates(name: str, entity_type: str, candidates: list[dict]) -> str | None:
    """LLM-суд: принимает имя+тип новой сущности и обогащённых кандидатов (см.
    `_describe_candidate`), возвращает ключ кандидата, признанного той же сущностью, либо
    `None`. Любой невалидный ответ (не JSON, отсутствующее поле, ключ не из списка кандидатов)
    трактуется как `None`, а не как ошибка — LLM не должна уметь уронить сборку графа."""
    if not candidates:
        return None

    valid_keys = {c["key"] for c in candidates}
    candidates_desc = "\n\n".join(
        f"- ключ: {c['key']}\n  имя: {c['name']}\n  тип: {c['type']}\n"
        f"  известные алиасы: {c['aliases']}\n  факты: {c['facts']}"
        for c in candidates
    )
    user_content = (
        f"НОВАЯ сущность:\nимя: {name}\nтип: {entity_type}\n\n"
        f"КАНДИДАТЫ из графа:\n{candidates_desc}"
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        raw = chat_complete(
            messages,
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=_JUDGE_MAX_TOKENS,
        )
        parsed = json.loads(raw)
    except Exception:
        # Обрезанный/пустой ответ (LLMResponseError), не-JSON, сетевая ошибка после
        # исчерпания ретраев — всё это трактуем как "не нашли", не роняем build_graph.
        return None

    if not isinstance(parsed, dict):
        return None
    same_as = parsed.get("same_as")
    if same_as in valid_keys:
        return same_as
    return None


def resolve_entity_semantically(
    G: nx.MultiDiGraph,
    entity_index: EntityIndex,
    name: str,
    entity_type: str,
) -> str | None:
    """Главная функция модуля: пытается найти в графе существующий узел, обозначающий ту же
    реальную сущность, что и `name` (тип `entity_type`), через embedding+FAISS+LLM.

    Возвращает ключ существующего узла, если LLM решила, что это тот же объект, иначе `None`
    (вызывающий код должен трактовать это как "новая сущность" — создать узел и добавить его
    эмбеддинг в `entity_index` через `add_to_index`, это НЕ делается здесь)."""
    candidate_keys = find_candidates(entity_index, name)
    if not candidate_keys:
        return None
    candidates = [_describe_candidate(G, key) for key in candidate_keys]
    return judge_candidates(name, entity_type, candidates)
