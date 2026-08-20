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

# Порог "похожести достаточно, чтобы спрашивать LLM без поверхностных улик" — см.
# `_worth_asking_llm`. Настоящие синонимы без общих букв ("britain"/"uk") живут выше него.
HIGH_SIMILARITY = 0.80

# Кандидату хватает пары фактов, чтобы LLM могла понять контекст, не раздувая промпт.
_MAX_FACTS_PER_CANDIDATE = 4
_JUDGE_MAX_TOKENS = 80

# Служебные слова, которые не дают буквы в аббревиатуру: IAAF — это International
# Association *of* Athletics Federations, "of" в акроним не попадает.
_ACRONYM_STOPWORDS = frozenset({"of", "the", "and", "for", "a", "an", "de", "in", "on"})


@dataclass
class EntityIndex:
    """Инкрементальный FAISS-индекс имён узлов графа: `index` хранит только векторы
    (L2-нормализованные, cosine similarity через inner product), `keys[i]` — ключ узла
    графа, отвечающий вектору на позиции `i`, `names[i]` — имя, которым этот узел был
    проиндексирован (FAISS сам метаданные не хранит, а имя нужно blocking-фильтру в
    `_worth_asking_llm`, чтобы не тащить в него весь граф)."""

    index: faiss.IndexFlatIP
    keys: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)


# Ставится в True после первого же отказа провайдера: без него каждая из сотен сущностей
# заново ждала бы таймаут и ретраи tenacity, растягивая "быстрое падение" на часы.
# Сбрасывается в `create_index`, то есть живёт ровно одну сборку графа, а не весь процесс.
_embedding_failed = False


def create_index() -> EntityIndex:
    """Пустой FAISS-индекс нужной размерности. Размерность берётся реальным вызовом
    `embed_texts`, а не хардкодится — она зависит от настроенной embedding-модели
    (для bge-m3 сейчас 1024, но это не должно быть magic-числом в коде). Если провайдер
    недоступен — индекс остаётся нулевой размерности и в него ничего не добавится, то есть
    резолюция мягко выродится в токенную эвристику (см. `_embed_name`)."""
    global _embedding_failed
    # Каждая сборка графа начинает с чистого листа: если прошлая упала из-за лежащего
    # провайдера, это не должно молча отключать резолюцию в следующей (в том же процессе).
    _embedding_failed = False
    probe = _embed_name("dimension probe")
    dim = int(probe.shape[1]) if probe is not None else 1
    return EntityIndex(index=faiss.IndexFlatIP(dim))


def _embed_name(name: str) -> np.ndarray | None:
    """L2-нормализованный вектор имени, либо `None`, если embedding-провайдер недоступен.

    Семантическая резолюция — улучшение поверх токенной эвристики, а не обязательный шаг:
    если LLM-провайдер лежит (или у ревьюера он вообще не сконфигурирован), сборка графа
    должна деградировать до токенной эвристики с предупреждением, а не падать трейсбеком
    на последнем шаге пайплайна, потеряв всю проделанную работу. Ровно так же ведёт себя
    и LLM-суд (см. `judge_candidates`)."""
    global _embedding_failed
    if _embedding_failed:
        return None
    try:
        return l2_normalize(embed_texts([name]).astype(np.float32))
    except Exception as exc:  # noqa: BLE001 — любая ошибка провайдера, не только сетевая
        _embedding_failed = True
        print(
            f"  [entity_resolution] эмбеддинги недоступны ({exc}); семантическая резолюция "
            "сущностей отключена до конца прогона, остаётся только токенная эвристика."
        )
        return None


def add_to_index(entity_index: EntityIndex, key: str, name: str) -> None:
    """Эмбеддит `name` и добавляет вектор в индекс, привязывая его к ключу узла `key`.
    Если эмбеддинги недоступны — тихо пропускает (см. `_embed_name`)."""
    vector = _embed_name(name)
    if vector is None:
        return
    entity_index.index.add(vector)
    entity_index.keys.append(key)
    entity_index.names.append(name)


def find_candidates(
    entity_index: EntityIndex,
    name: str,
    *,
    top_k: int = TOP_K_CANDIDATES,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[str]:
    """Ключи узлов-кандидатов: топ-k ближайших соседей `name` в индексе по косинусному
    сходству, отфильтрованные по `threshold` и по blocking-правилу `_worth_asking_llm`
    (оно отсеивает пары, на которых LLM заведомо нечего решать — см. его докстринг).
    Никаких обращений к LLM здесь — чисто векторный поиск, LLM-суд — отдельный шаг
    (`judge_candidates`/`resolve_entity_semantically`)."""
    if entity_index.index.ntotal == 0:
        return []
    vector = _embed_name(name)
    if vector is None:
        return []
    k = min(top_k, entity_index.index.ntotal)
    scores, indices = entity_index.index.search(vector, k)
    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < threshold:
            continue
        if not _worth_asking_llm(name, entity_index.names[idx], float(score)):
            continue
        candidates.append(entity_index.keys[idx])
    return candidates


def _name_tokens(name: str) -> list[str]:
    """Токены имени без обрамляющей пунктуации: "(BNP)" -> "bnp", "Barstow," -> "barstow"."""
    stripped = (t.strip("()[].,:;\"'").casefold() for t in name.split())
    return [t for t in stripped if t]


def _acronym_of(tokens: list[str]) -> str:
    return "".join(t[0] for t in tokens if t and t not in _ACRONYM_STOPWORDS)


def has_surface_evidence(a: str, b: str) -> bool:
    """Есть ли у двух имён поверхностная улика, что это формы ОДНОГО имени:
    вложенность токенов ("amnesty" в "amnesty international", "bnp" в "british national
    party (bnp)") либо аббревиатура ("iaaf" = International Association of Athletics
    Federations, "eu" = European Union).

    Аббревиатуру признаём только у многословного имени: иначе односложное "3" сошло бы
    за аббревиатуру "3ami" (реальный ложный случай с этого корпуса)."""
    a_tokens, b_tokens = _name_tokens(a), _name_tokens(b)
    if not a_tokens or not b_tokens:
        return False
    if set(a_tokens) <= set(b_tokens) or set(b_tokens) <= set(a_tokens):
        return True
    for short, long_form in ((a_tokens, b_tokens), (b_tokens, a_tokens)):
        if len(short) == 1 and len(long_form) >= 2 and short[0] == _acronym_of(long_form):
            return True
    return False


def _worth_asking_llm(name: str, candidate_name: str, score: float) -> bool:
    """Стоит ли тратить LLM-вызов на пару "новая сущность / кандидат".

    Это blocking-шаг, стандартный для entity resolution: дешёвый фильтр перед дорогим
    сравнением. Нужен по двум причинам, обе подтверждены замерами на этом корпусе:

    1. Точность. В зоне похожести 0.55-0.80 у маленькой локальной модели нет ни одной
       зацепки, кроме самих имён, и она начинает угадывать: на прогоне без этого фильтра
       она слила футболиста "cisse" с "nicolas cage" (0.559), "ebell" с "ebay" (0.666),
       "bill" с "bush" (0.714) — то самое ложное слияние, которое портит граф сильнее,
       чем недомерж. С фильтром 10 из 13 таких ложных слияний не доходят до LLM вообще.
    2. Цена. Без фильтра LLM-суд звался 704 раза на 100 документов и занимал часы на
       локальной модели — больше, чем вся остальная сборка графа вместе взятая.

    Пропускаем пару, если есть поверхностная улика (см. `has_surface_evidence`) ЛИБО
    похожесть настолько высока, что улики не нужны: "britain"/"uk" (0.837) и
    "tories"/"conservatives" — настоящие синонимы без единой общей буквы, и первый из них
    проходит именно по этому правилу. Второй — цена фильтра, см. README."""
    return score >= HIGH_SIMILARITY or has_surface_evidence(name, candidate_name)


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
        "name": data.get("name", key),
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
