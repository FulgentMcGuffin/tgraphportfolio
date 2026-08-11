"""End-to-end notebook-equivalent pipeline for a single network view."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from .config import PipelineConfig
from .data_access import load_table
from .measures import compute_measure
from .network import build_corr_nx, pivot_to_wide
from .pyvis_plot import graph_to_html
from .transforms import apply_transforms

ProgressCallback = Callable[[int, int, str], None]
StatusCallback = Callable[[str], None]


def _needed_columns(cfg: PipelineConfig) -> list[str]:
    cols = [cfg.date_column, cfg.name_column, cfg.value_column]
    if cfg.filter_column:
        cols.append(cfg.filter_column)
    return list(dict.fromkeys(cols))


def _prepare_frame(cfg: PipelineConfig, status: StatusCallback | None = None) -> pl.DataFrame:
    needed = _needed_columns(cfg)
    df = load_table(cfg.db_path, cfg.table, columns=needed)

    df = df.with_columns(pl.col(cfg.date_column).cast(pl.Date, strict=False))
    df = df.sort(cfg.date_column, cfg.name_column)

    if cfg.filter_column and cfg.filter_value is not None:
        if status is not None:
            status(
                f"Filtering {cfg.filter_column} = {cfg.filter_value!r}…"
            )
        df = df.filter(pl.col(cfg.filter_column).cast(pl.Utf8) == cfg.filter_value)

    if cfg.date_start is not None:
        df = df.filter(pl.col(cfg.date_column) >= cfg.date_start)
    if cfg.date_end is not None:
        df = df.filter(pl.col(cfg.date_column) <= cfg.date_end)

    df = df.drop_nulls(subset=[cfg.date_column, cfg.name_column, cfg.value_column])

    transform_cols = [cfg.value_column]
    if cfg.transforms:
        if status is not None:
            status(f"Applying transforms: {', '.join(cfg.transforms)}…")
        df = apply_transforms(
            df,
            cfg.transforms,
            cfg.date_column,
            cfg.name_column,
            transform_cols,
        )

    return df


def run_pipeline(
    cfg: PipelineConfig,
    *,
    progress: ProgressCallback | None = None,
    status: StatusCallback | None = None,
) -> str:
    """Run the full pipeline and return pyvis HTML."""

    def _status(msg: str) -> None:
        if status is not None:
            status(msg)

    _status("Loading data…")
    df = _prepare_frame(cfg, status=status)
    if df.is_empty():
        raise ValueError("No rows remain after filtering / date range / transforms.")
    _status(f"Loaded {df.height:,} rows after filters/transforms.")

    _status(
        f"Pivoting wide on {cfg.value_column!r} "
        f"(date={cfg.date_column!r}, node={cfg.name_column!r})…"
    )
    wide = pivot_to_wide(
        df,
        date_column=cfg.date_column,
        name_column=cfg.name_column,
        value_column=cfg.value_column,
    )
    nodes = [c for c in wide.columns if c != cfg.date_column]
    if len(nodes) < 2:
        raise ValueError("Need at least two nodes to build a network.")
    _status(f"Wide matrix: {wide.height:,} dates × {len(nodes)} nodes.")

    n_pairs = sum(len(nodes[k:]) for k in range(len(nodes)))
    _status(f"Computing {cfg.measure} ({n_pairs:,} pairs)…")
    measure_df = compute_measure(
        cfg.measure, wide.select(nodes), nodes, progress=progress
    )

    _status(
        f"Building network (independence threshold={cfg.independent_threshold:.2f})…"
    )
    graph = build_corr_nx(
        measure_df, independent_threshold=cfg.independent_threshold
    )
    _status(
        f"Network built: |N|={graph.number_of_nodes()}, "
        f"|E|={graph.number_of_edges():,}."
    )

    _status("Rendering interactive network…")
    html = graph_to_html(graph, title=cfg.title)
    _status("Done.")
    return html
