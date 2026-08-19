"""Извлечение сущностей и связей из документов через LLM.

Один документ — один LLM-вызов с запросом строгого JSON (сущности + триплеты
subject/predicate/object). Каждая связь хранит doc_id, откуда она взята — это то, что
позже позволяет графу отвечать не только "кто с кем связан", но и "в каком документе
это упомянуто", а RAG-модулю — цитировать источник факта.

Экстракция на документ, а не на весь датасет разом, потому что: (1) контекст короче и
предсказуемее для LLM, меньше шанс упустить сущности из середины батча; (2) источник
факта получается бесплатно (doc_id уже известен на входе); (3) кэш в llm_client работает на
уровне отдельного запроса, так что добавление новых документов не требует повторной
экстракции старых.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from graph_rag.config import settings
from graph_rag.ingest import load_saved_corpus
from graph_rag.llm_client import chat_complete

EXTRACTIONS_PATH = settings.artifacts_dir / "extractions.json"

_SYSTEM_PROMPT = """\
Ты извлекаешь сущности и связи между ними из новостного текста для построения графа знаний.

Сущности: люди, организации, места, продукты — только те, что явно упомянуты в тексте.
Имя сущности — короткое настоящее имя (обычно 1-6 слов), как "Tony Blair" или
"International Association of Athletics Federations", а НЕ фраза и не предложение из
текста. Если для места в тексте нет короткого собственного имени — не создавай из него
сущность.

Связи: тройки (subject, predicate, object), где subject и object — сущности из списка entities
(с тем же ограничением на длину имени), а predicate — короткая глагольная фраза на английском
(например "works_for", "located_in", "acquired", "met_with"). Постарайся дать хотя бы одну
связь для каждой сущности из списка entities, если из текста можно вывести любую осмысленную
связь (в том числе слабую, например "mentioned_in", "participated_in") — не оставляй сущность
совсем без связей, если её роль в тексте хоть как-то понятна.

Ответь СТРОГО валидным JSON без пояснений, в формате:
{"entities": [{"name": "...", "type": "person|organization|location|product|other"}],
 "relations": [{"subject": "...", "predicate": "...", "object": "..."}]}

Если сущностей или связей нет — верни пустые списки. Имена сущностей — в исходном
написании из текста, без сокращений и добавленных пояснений."""


@dataclass
class DocExtraction:
    doc_id: str
    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)


# Промпт просит "имена сущностей в исходном написании, без сокращений и добавленных
# пояснений", но LLM это не всегда соблюдает — иногда вместо короткого имени в subject/
# object триплета или в entities попадает целая фраза/предложение из текста статьи
# (например, "52% rise in profits for the year to £198m from the £130m seen a year
# earlier" вместо настоящей сущности). Порог в 6 слов подобран так, чтобы не резать
# легитимные длинные названия организаций/изданий (например, "international association
# of athletics federations" — 6 слов), но отсечь клаузы/предложения — на реальных данных
# датасета 7+ слов почти без исключений оказывались мусором, а не именами.
_MAX_ENTITY_NAME_WORDS = 6


def is_valid_entity_name(name: str) -> bool:
    """True, если `name` похоже на короткое имя сущности, а не на фразу/предложение
    (см. `_MAX_ENTITY_NAME_WORDS`). Используется и здесь при извлечении (фильтрует
    будущие LLM-вызовы), и в graph_store.build_graph (чистит уже закэшированные
    extractions.json задним числом, без повторного вызова LLM)."""
    name = name.strip()
    if not name:
        return False
    if name.endswith((".", "!", "?")):
        return False
    return len(name.split()) <= _MAX_ENTITY_NAME_WORDS


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def extract_from_text(doc_id: str, text: str) -> DocExtraction:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    try:
        raw = chat_complete(messages, response_format={"type": "json_object"}, max_tokens=1200)
    except Exception:
        raw = chat_complete(messages, max_tokens=1200)

    try:
        data = _parse_json(raw)
    except (json.JSONDecodeError, AttributeError):
        return DocExtraction(doc_id=doc_id)

    entities = [
        e
        for e in data.get("entities", [])
        if isinstance(e, dict) and e.get("name") and is_valid_entity_name(e["name"])
    ]
    relations = [
        r
        for r in data.get("relations", [])
        if isinstance(r, dict)
        and r.get("subject")
        and r.get("object")
        and r.get("predicate")
        and is_valid_entity_name(r["subject"])
        and is_valid_entity_name(r["object"])
    ]
    return DocExtraction(doc_id=doc_id, entities=entities, relations=relations)


def build_extractions(force: bool = False) -> list[dict]:
    if EXTRACTIONS_PATH.exists() and not force:
        return json.loads(EXTRACTIONS_PATH.read_text(encoding="utf-8"))

    docs = load_saved_corpus()
    results = []
    total = len(docs)
    for i, (_, row) in enumerate(docs.iterrows(), start=1):
        print(f"  [{i}/{total}] {row['doc_id']}", flush=True)
        extraction = extract_from_text(row["doc_id"], row["text"])
        results.append(
            {
                "doc_id": extraction.doc_id,
                "entities": extraction.entities,
                "relations": extraction.relations,
            }
        )

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    EXTRACTIONS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def load_extractions() -> list[dict]:
    if not EXTRACTIONS_PATH.exists():
        raise FileNotFoundError(f"{EXTRACTIONS_PATH} не найден — сначала прогони build_extractions().")
    return json.loads(EXTRACTIONS_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    data = build_extractions()
    n_entities = sum(len(d["entities"]) for d in data)
    n_relations = sum(len(d["relations"]) for d in data)
    print(f"{len(data)} документов -> {n_entities} сущностей (с повторами), {n_relations} связей")
