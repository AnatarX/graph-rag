"""Загрузка сырого корпуса и нормализация в единую схему датасета.

Датасет выбирается через `settings.dataset` — ключ реестра `DATASET_LOADERS`. Загрузчик
знает только про свой источник и возвращает "сырой" DataFrame с обязательной колонкой
`text` и необязательными `category`/`title`; всё остальное (сэмплирование, генерация
заголовков, doc_id, профиль корпуса) делает общий код ниже. Добавить новый датасет =
написать функцию без аргументов и вписать её в реестр.

Схема документа на выходе: doc_id, category (если у датасета она есть — не то же самое,
что кластеры из `clustering.py`, но полезный sanity-check), title, text.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from graph_rag.config import settings

TITLE_WORD_COUNT = 12

DOCS_PATH = settings.dataset_artifacts_dir / "docs.parquet"
CORPUS_PROFILE_PATH = settings.dataset_artifacts_dir / "corpus_profile.json"


# --- загрузчики датасетов ---------------------------------------------------


def load_bbc() -> pd.DataFrame:
    """BBC News CSV (см. scripts/download_data.py): колонки `category` и `text`,
    ~2225 статей по 5 категориям. Собственных заголовков в датасете нет."""
    csv_files = sorted(settings.data_raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Нет .csv в {settings.data_raw_dir}. Сначала запусти "
            "`uv run python scripts/download_data.py`."
        )
    raw = pd.read_csv(csv_files[0])
    missing = {"category", "text"} - set(raw.columns)
    if missing:
        raise ValueError(
            f"В {csv_files[0].name} нет колонок {sorted(missing)} — ожидается BBC News CSV "
            "с колонками `category`/`text`."
        )
    return raw[["category", "text"]]


# 20 Newsgroups после remove=("headers","footers","quotes") наполовину состоит из
# огрызков: пустые сообщения, одна строка подписи, "Me too." — из таких документов
# нечего извлекать (ни сущностей, ни связей), а эмбеддинги у них шумные. Порог по длине
# оставляет 5571 документ из 11314 — этого с запасом хватает на выборку в n_docs.
_20NG_MIN_DOC_CHARS = 500


def load_20newsgroups() -> pd.DataFrame:
    """20 Newsgroups из sklearn (скачивается и кэшируется самим sklearn при первом
    обращении). Второй датасет нужен именно как непохожий на BBC: нормальный регистр
    букв, другая длина документов, 20 категорий вместо 5 — то есть проверка, что
    пайплайн не заточен под один корпус."""
    from sklearn.datasets import fetch_20newsgroups

    bunch = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    raw = pd.DataFrame(
        {
            "text": [t.strip() for t in bunch.data],
            "category": [bunch.target_names[t] for t in bunch.target],
        }
    )
    return raw[raw["text"].str.len() >= _20NG_MIN_DOC_CHARS].reset_index(drop=True)


# Загрузчик — функция без аргументов, возвращающая DataFrame с обязательной колонкой
# `text` и опциональными `category` (включает стратифицированное сэмплирование) и
# `title` (иначе заголовок генерируется из первых слов текста).
DATASET_LOADERS: dict[str, Callable[[], pd.DataFrame]] = {
    "bbc": load_bbc,
    "20newsgroups": load_20newsgroups,
}


def get_loader(dataset: str | None = None) -> Callable[[], pd.DataFrame]:
    dataset = dataset or settings.dataset
    try:
        return DATASET_LOADERS[dataset]
    except KeyError:
        raise ValueError(
            f"Неизвестный датасет {dataset!r}. Доступные: {', '.join(sorted(DATASET_LOADERS))}. "
            "Задаётся через DATASET=... в .env или переменной окружения."
        ) from None


# --- нормализация схемы -----------------------------------------------------


def _make_title(text: str) -> str:
    words = text.split()
    title = " ".join(words[:TITLE_WORD_COUNT])
    return title[0].upper() + title[1:] if title else ""


def load_corpus(n_docs: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Выборка из датасета `settings.dataset` + нормализация схемы.

    Если у датасета есть колонка `category`, выборка стратифицированная — так структура
    датасета сохраняется в уменьшенном виде (и кластеризацию потом есть с чем сверить).
    Если категорий нет — обычная случайная выборка: стратифицировать не по чему, и это
    единственная разница в поведении между "богатым" и "голым" датасетом.
    """

    n_docs = n_docs or settings.n_docs
    seed = seed if seed is not None else settings.random_seed

    raw = get_loader()()
    if "text" not in raw.columns:
        raise ValueError(
            f"Загрузчик датасета {settings.dataset!r} не вернул обязательную колонку `text` "
            f"(есть: {sorted(raw.columns)})."
        )

    subset = [c for c in ("category", "text") if c in raw.columns]
    raw = raw.dropna(subset=subset).reset_index(drop=True)
    if raw.empty:
        raise ValueError(f"Датасет {settings.dataset!r} пуст после отбрасывания пустых значений.")

    frac = min(1.0, n_docs / len(raw))
    if "category" in raw.columns:
        sampled = raw.groupby("category", group_keys=False).sample(frac=frac, random_state=seed)
    else:
        sampled = raw.sample(frac=frac, random_state=seed)

    # Округление доли внутри каждой группы может недобрать до n_docs, и тем сильнее, чем
    # больше категорий: у BBC (5 категорий) выходит ровно 100 документов, у 20 Newsgroups
    # (20 категорий) — 98. Добираем недостающее случайно из остатка, чтобы n_docs означал
    # одно и то же на любом датасете. Для BBC ветка не срабатывает — выборка та же, что и
    # до появления этого кода.
    shortfall = n_docs - len(sampled)
    if shortfall > 0:
        rest = raw.drop(index=sampled.index)
        if not rest.empty:
            sampled = pd.concat([sampled, rest.sample(n=min(shortfall, len(rest)), random_state=seed)])

    sampled = (
        sampled.sample(frac=1.0, random_state=seed)  # перемешать порядок между категориями
        .reset_index(drop=True)
        .iloc[:n_docs]
        .reset_index(drop=True)
    )

    # Префикс doc_id — имя датасета: doc_id уезжает в extractions.json, в граф и в ответы
    # LLM ("[bbc-0035]"), поэтому "bbc-0001" и "20newsgroups-0001" не должны быть одним и
    # тем же идентификатором даже при случайном смешении артефактов.
    sampled["doc_id"] = [f"{settings.dataset}-{i:04d}" for i in range(len(sampled))]
    if "title" not in sampled.columns:
        sampled["title"] = sampled["text"].map(_make_title)
    columns = ["doc_id"] + (["category"] if "category" in sampled.columns else []) + ["title", "text"]
    return sampled[columns]


# --- профиль корпуса --------------------------------------------------------

# Доля документов, содержащих хоть одну заглавную букву, выше которой считаем, что
# регистр в корпусе несёт информацию. Порог грубый намеренно: реальные корпуса ложатся
# по краям (BBC News CSV — 0.0, во всём корпусе нет ни одной заглавной; 20 Newsgroups —
# практически 1.0), промежуточных значений на практике не бывает.
_MIN_DOCS_WITH_UPPERCASE = 0.5
# Вторая, независимая проверка: доля заглавных среди всех букв. Отсекает вырожденный
# случай "текст в нижнем регистре, но в каждом документе есть одинокая заглавная"
# (артефакт конвертации, инициал, номер версии) — на нормальной английской прозе доля
# заглавных ~2-4%, так что 0.5% — с большим запасом ниже.
_MIN_UPPERCASE_LETTERS = 0.005


def compute_corpus_profile(texts: Iterable[str]) -> dict:
    """Характеристики корпуса, от которых зависят эвристики пайплайна.

    Считается один раз при ingest и сохраняется рядом с docs.parquet, чтобы шаги ниже
    по течению (экстракция сущностей, длина сниппета) подстраивались под датасет, а не
    под константы, подобранные под BBC:

    * `has_meaningful_case` — можно ли использовать капитализацию как сигнал
      "это имя собственное" (см. `graph_extraction.is_valid_entity_name`). В BBC News CSV
      текст приходит целиком в нижнем регистре, и такой фильтр там резал бы легитимные
      сущности наравне с мусором; в корпусе с нормальным регистром — наоборот, это
      сильный и дешёвый фильтр.
    * `median_doc_chars` — типичная длина документа, из неё считается размер сниппета
      в `rag` (у BBC медиана ~2.1k символов, у 20 Newsgroups — почти на порядок меньше,
      и фиксированные 2200 там просто не имеют смысла).
    """
    lengths: list[int] = []
    docs_with_upper = 0
    letters = 0
    upper_letters = 0

    for text in texts:
        text = str(text)
        lengths.append(len(text))
        has_upper = False
        for ch in text:
            if ch.isalpha():
                letters += 1
                if ch.isupper():
                    upper_letters += 1
                    has_upper = True
        docs_with_upper += has_upper

    n_docs = len(lengths)
    docs_with_upper_ratio = docs_with_upper / n_docs if n_docs else 0.0
    upper_letters_ratio = upper_letters / letters if letters else 0.0

    return {
        "n_docs": n_docs,
        "median_doc_chars": int(statistics.median(lengths)) if lengths else None,
        "docs_with_uppercase_ratio": round(docs_with_upper_ratio, 4),
        "uppercase_letters_ratio": round(upper_letters_ratio, 4),
        "has_meaningful_case": (
            docs_with_upper_ratio > _MIN_DOCS_WITH_UPPERCASE
            and upper_letters_ratio > _MIN_UPPERCASE_LETTERS
        ),
    }


# Фолбэк = поведение до появления профиля: капитализация не используется как сигнал,
# длина сниппета берётся из константы rag.SNIPPET_CHARS. Так старые артефакты (собранные
# до этого файла) и любой вызов без ingest-шага продолжают работать, а не падают.
_PROFILE_FALLBACK = {"has_meaningful_case": False, "median_doc_chars": None}


def load_corpus_profile(path: Path | None = None) -> dict:
    path = path if path is not None else CORPUS_PROFILE_PATH
    if not path.exists():
        return dict(_PROFILE_FALLBACK)
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(_PROFILE_FALLBACK)
    return {**_PROFILE_FALLBACK, **profile}


# --- persist ----------------------------------------------------------------


def save_corpus(df: pd.DataFrame) -> None:
    settings.dataset_artifacts_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DOCS_PATH, index=False)
    _save_corpus_profile(df)


def _save_corpus_profile(df: pd.DataFrame) -> dict:
    profile = compute_corpus_profile(df["text"])
    CORPUS_PROFILE_PATH.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def ensure_corpus_profile(df: pd.DataFrame) -> dict:
    """Досчитывает профиль корпуса, если его ещё нет рядом с docs.parquet.

    Нужно для артефактов, собранных до появления профиля: `pipeline build` пропускает
    шаг ingest по наличию docs.parquet, поэтому без этого corpus_profile.json для уже
    существующего корпуса не появился бы никогда (только через --force, то есть с
    полным пересчётом эмбеддингов и графа)."""
    if CORPUS_PROFILE_PATH.exists():
        return load_corpus_profile()
    settings.dataset_artifacts_dir.mkdir(parents=True, exist_ok=True)
    return _save_corpus_profile(df)


def load_saved_corpus() -> pd.DataFrame:
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"{DOCS_PATH} не найден — сначала прогони пайплайн ingest.")
    return pd.read_parquet(DOCS_PATH)


if __name__ == "__main__":
    corpus = load_corpus()
    save_corpus(corpus)
    print(f"Сохранено {len(corpus)} документов ({settings.dataset}) в {DOCS_PATH}")
    if "category" in corpus.columns:
        print(corpus["category"].value_counts())
    print(f"Профиль корпуса -> {CORPUS_PROFILE_PATH}: {load_corpus_profile()}")
