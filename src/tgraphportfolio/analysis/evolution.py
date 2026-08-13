"""Temporal evolution of network metrics via rolling windows."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date

import networkx as nx
import numpy as np
import polars as pl

from . import measures, network, transforms

ProgressCallback = Callable[[int, int, str], None]
StatusCallback = Callable[[str], None]


@dataclass
class EvolutionConfig:
    """Configuration for rolling/expanding-window network evolution analysis.

    Attributes:
        window_size: Number of observations per window (trading days, default 252 ~1 year).
        step: Observations to advance between windows (default 21 ~1 month).
        expanding: If True, anchor window start at first date (expanding window mode).
        min_nodes: Minimum nodes required per window to include it (default 5).
        independent_threshold: Correlation threshold for network edges (default 0.33).
        centrality: Per-node centrality measure: "eigenvector", "betweenness", or "degree".
        n_top_nodes: Number of top variable nodes to show in centrality plot (default 10).
    """

    window_size: int = 252
    step: int = 21
    expanding: bool = False
    min_nodes: int = 5
    independent_threshold: float = 0.33
    centrality: str = "eigenvector"
    n_top_nodes: int = 10


def generate_windows(
    dates: list[date],
    window_size: int,
    step: int,
    *,
    expanding: bool = False,
) -> Iterator[tuple[date, date, list[date]]]:
    """Yield rolling or expanding windows over sorted unique trading dates.

    Args:
        dates: Sorted, unique trading dates (ascending).
        window_size: Minimum/initial number of observations per window.
        step: Number of observations to advance between windows.
        expanding: Anchor window start at dates[0] instead of sliding it.

    Yields:
        (window_start, window_end, window_dates) tuples.

    Raises:
        ValueError: If window_size < 3.
    """
    if window_size < 3:
        raise ValueError("window_size must be >= 3")
    n = len(dates)
    end_idx = window_size - 1
    while end_idx < n:
        start_idx = 0 if expanding else end_idx - window_size + 1
        window_dates = dates[start_idx : end_idx + 1]
        yield window_dates[0], window_dates[-1], window_dates
        end_idx += step


def _drop_nan_edges(graph: nx.Graph) -> nx.Graph:
    """Remove NaN-weight edges."""
    graph = graph.copy()
    bad_edges = [(u, v) for u, v, w in graph.edges(data="weight") if w != w]
    graph.remove_edges_from(bad_edges)
    return graph


def _add_strength_attr(graph: nx.Graph) -> None:
    """Attach strength attribute (1 - weight) in place."""
    for _, _, d in graph.edges(data=True):
        d["strength"] = 1.0 - d["weight"]


def _node_centrality(graph: nx.Graph, centrality: str) -> dict[str, float]:
    """Compute per-node centrality, robust to disconnected graphs."""
    if graph.number_of_edges() == 0:
        return dict.fromkeys(graph.nodes(), 0.0)
    if centrality == "eigenvector":
        try:
            return nx.eigenvector_centrality(graph, weight="strength", max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            return nx.betweenness_centrality(graph, weight="weight")
    if centrality == "betweenness":
        return nx.betweenness_centrality(graph, weight="weight")
    if centrality == "degree":
        return nx.degree_centrality(graph)
    raise ValueError(f"Unknown centrality: {centrality!r}")


def _node_summary(
    graph: nx.Graph, window_end: date, *, centrality: str
) -> list[dict]:
    """Long-format per-node metric rows."""
    cent = _node_centrality(graph, centrality)
    weighted_degree = dict(graph.degree(weight="strength"))
    rows = []
    for node in graph.nodes():
        rows.append(
            {
                "window_end": window_end,
                "node": node,
                "metric": "weighted_degree",
                "value": float(weighted_degree[node]),
            }
        )
        rows.append(
            {
                "window_end": window_end,
                "node": node,
                "metric": centrality,
                "value": float(cent.get(node, float("nan"))),
            }
        )
    return rows


def compute_evolution_metrics(
    df_returns: pl.DataFrame,
    dates: list[date],
    cfg: EvolutionConfig,
    *,
    date_column: str = "Date",
    name_column: str = "Name",
    value_column: str = "Close",
    progress: ProgressCallback | None = None,
    status: StatusCallback | None = None,
) -> pl.DataFrame:
    """Compute node-level metrics (weighted_degree, centrality) for every window.

    Args:
        df_returns: Long-format daily-returns dataframe.
        dates: Sorted unique trading dates.
        cfg: Evolution configuration.
        date_column, name_column, value_column: Column names.
        progress: Progress callback (done, total, desc).
        status: Status message callback.

    Returns:
        Node metrics DataFrame (long format, one row per window/node/metric).
    """
    if status is not None:
        status(f"Generating {cfg.window_size}-obs windows (step={cfg.step})...")

    windows = list(
        generate_windows(dates, cfg.window_size, cfg.step, expanding=cfg.expanding)
    )
    node_rows = []
    n_windows = len(windows)

    for w_idx, (window_start, window_end, window_dates) in enumerate(windows):
        if progress is not None:
            progress(w_idx, n_windows, f"{window_end}")

        window_df = df_returns.filter(
            pl.col(date_column).is_between(window_start, window_end)
        )
        wide = network.pivot_to_wide(
            window_df, date_column, name_column, value_column
        )
        nodes = [c for c in wide.columns if c != date_column]
        nodes = [n for n in nodes if wide.get_column(n).drop_nulls().len() >= 3]

        if len(nodes) < cfg.min_nodes:
            continue

        measure_df = measures.compute_measure(
            "distance_correlation", wide.select(nodes), nodes
        )
        graph = network.build_corr_nx(
            measure_df, independent_threshold=cfg.independent_threshold
        )
        graph = _drop_nan_edges(graph)
        _add_strength_attr(graph)
        node_rows.extend(
            _node_summary(graph, window_end, centrality=cfg.centrality)
        )

    if progress is not None:
        progress(n_windows, n_windows, "done")

    return pl.DataFrame(node_rows)


def get_top_variable_nodes(
    node_metrics: pl.DataFrame, metric: str, k: int = 10
) -> list[str]:
    """Return k node names with highest across-window std for metric."""
    return (
        node_metrics.filter(pl.col("metric") == metric)
        .group_by("node")
        .agg(pl.col("value").std().alias("std"))
        .sort("std", descending=True)
        .head(k)
        .get_column("node")
        .to_list()
    )
