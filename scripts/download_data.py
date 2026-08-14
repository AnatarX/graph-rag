"""Скачивает датасет BBC News с Kaggle через kagglehub.

Датасет: yufengdev/bbc-fulltext-and-category — один CSV (bbc-text.csv) с колонками
`category` и `text`, ~2225 новостных статей по 5 категориям (business, entertainment,
politics, sport, tech). Берём его, а не более крупные датасета вроде полного BBC News
Summary, потому что для задания важна не полнота, а чистая структура: короткие
однозначные категории и плотный текст с именованными сущностями.

Требует Kaggle-аутентификации: переменные окружения KAGGLE_USERNAME/KAGGLE_KEY либо
файл ~/.kaggle/kaggle.json (Kaggle -> Settings -> Create New Token).

Запуск: `uv run python scripts/download_data.py`
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import kagglehub

from graph_rag.config import settings

DATASET_ID = "yufengdev/bbc-fulltext-and-category"


def main() -> Path:
    settings.data_raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        cache_path = Path(kagglehub.dataset_download(DATASET_ID))
    except Exception as exc:  # noqa: BLE001 — печатаем понятную инструкцию и падаем
        print(
            "Не удалось скачать датасет с Kaggle.\n"
            "Нужна Kaggle-аутентификация: создай токен на "
            "https://www.kaggle.com/settings -> API -> Create New Token,\n"
            "сохрани как ~/.kaggle/kaggle.json или задай переменные окружения "
            "KAGGLE_USERNAME / KAGGLE_KEY.\n"
            f"Оригинальная ошибка: {exc}",
            file=sys.stderr,
        )
        raise

    csv_files = list(cache_path.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"В {cache_path} не найдено ни одного .csv файла")

    dest = settings.data_raw_dir / csv_files[0].name
    shutil.copy(csv_files[0], dest)
    print(f"Датасет сохранён в {dest}")
    return dest


if __name__ == "__main__":
    main()
