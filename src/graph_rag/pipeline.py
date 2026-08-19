"""CLI: склеивает весь пайплайн построения GraphRAG-индекса в один проход.

ingest -> embeddings -> clustering -> graph_extraction -> graph_store

Каждый шаг кэшируется по наличию своего артефакта в `artifacts/` — повторный запуск
`build` без `--force` почти мгновенно пропускает уже сделанные шаги. Это важно при
итеративной разработке: отладка графа не должна пересчитывать эмбеддинги заново.

Запуск: `uv run python -m graph_rag.pipeline build [--force]`
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import typer

from graph_rag.clustering import CLUSTERS_PATH, build_clusters
from graph_rag.config import settings
from graph_rag.embeddings import EMBEDDINGS_PATH, build_embeddings
from graph_rag.graph_extraction import EXTRACTIONS_PATH, build_extractions
from graph_rag.graph_store import GRAPH_PATH, attach_clusters, build_graph, load_graph, save_graph
from graph_rag.ingest import load_corpus, load_saved_corpus, save_corpus

app = typer.Typer(add_completion=False)

DOCS_PATH = settings.artifacts_dir / "docs.parquet"
DOC_CLUSTERS_PATH = settings.artifacts_dir / "doc_clusters.parquet"


@app.command()
def build(force: bool = typer.Option(False, "--force", help="Пересчитать все шаги заново")) -> None:
    if force or not DOCS_PATH.exists():
        typer.echo("[1/5] ingest: сэмплирую датасет...")
        save_corpus(load_corpus())
    else:
        typer.echo("[1/5] ingest: уже есть, пропускаю")
    docs = load_saved_corpus()
    typer.echo(f"      {len(docs)} документов")

    if force or not EMBEDDINGS_PATH.exists():
        typer.echo("[2/5] embeddings: считаю через LLM-провайдера...")
        build_embeddings(force=force)
    else:
        typer.echo("[2/5] embeddings: уже есть, пропускаю")

    if force or not CLUSTERS_PATH.exists():
        typer.echo("[3/5] clustering: KMeans + LLM-нейминг...")
        clusters = build_clusters(force=force)
        typer.echo(f"      k={clusters['k']}")
    else:
        typer.echo("[3/5] clustering: уже есть, пропускаю")

    if force or not EXTRACTIONS_PATH.exists():
        typer.echo("[4/5] graph_extraction: LLM извлекает сущности/связи по документам...")
        build_extractions(force=force)
    else:
        typer.echo("[4/5] graph_extraction: уже есть, пропускаю")

    if force or not GRAPH_PATH.exists():
        typer.echo("[5/5] graph_store: собираю граф...")
        extractions = json.loads(EXTRACTIONS_PATH.read_text(encoding="utf-8"))
        G = build_graph(extractions)

        import pandas as pd

        doc_clusters = pd.read_parquet(DOC_CLUSTERS_PATH)
        attach_clusters(G, doc_clusters)
        save_graph(G)
        typer.echo(f"      {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")
    else:
        typer.echo("[5/5] graph_store: уже есть, пропускаю")
        G = load_graph()

    # Числа узлов/рёбер зависят от модели, которая извлекала граф (см. README,
    # раздел "Срезанные углы") — без привязки к модели и дате прогона они
    # невоспроизводимы и бесполезны для проверки. Пишем метаданные ВСЕГДА (а не
    # только в ветке свежей сборки) — иначе для graph.json, уже лежавшего на диске
    # до появления этого файла (или собранного через skip-ветку), run_metadata.json
    # никогда бы не появился без --force.
    run_metadata = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "chat_model": settings.llm_chat_model,
        "embed_model": settings.llm_embed_model,
        "n_docs": len(docs),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
    }
    metadata_path = settings.artifacts_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"      метаданные прогона -> {metadata_path}")

    typer.echo("Готово. Дальше: `uv run streamlit run app/streamlit_app.py`")


@app.command()
def status() -> None:
    artifacts = {
        "docs.parquet": DOCS_PATH,
        "embeddings.npy": EMBEDDINGS_PATH,
        "clusters.json": CLUSTERS_PATH,
        "doc_clusters.parquet": DOC_CLUSTERS_PATH,
        "extractions.json": EXTRACTIONS_PATH,
        "graph.json": GRAPH_PATH,
    }
    for name, path in artifacts.items():
        mark = "✓" if path.exists() else "·"
        typer.echo(f"{mark} {name}")


if __name__ == "__main__":
    app()
