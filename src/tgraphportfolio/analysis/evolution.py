"""Temporal evolution of network metrics via rolling windows."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date

from graspologic.cluster import KMeansCluster
from graspologic.embed import AdjacencySpectralEmbed
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
        max_communities: Upper bound for per-window community count (ASE+KMeans, default 10).
    """

    window_size: int = 252
    step: int = 21
    expanding: bool = False
    min_nodes: int = 5
    independent_threshold: float = 0.33
    centrality: str = "eigenvector"
    n_top_nodes: int = 10
    max_communities: int = 10


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


def _network_summary(graph: nx.Graph, window_start: date, window_end: date) -> dict:
    """One row of network-level summary metrics for a window's graph."""
    n_nodes = graph.number_of_nodes()
    degrees = [d for _, d in graph.degree()]
    largest_cc = max((len(c) for c in nx.connected_components(graph)), default=0)
    return {
        "window_start": window_start,
        "window_end": window_end,
        "n_nodes": n_nodes,
        "n_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "avg_degree": (sum(degrees) / n_nodes) if n_nodes else float("nan"),
        "n_components": nx.number_connected_components(graph) if n_nodes else 0,
        "largest_component_size": largest_cc,
        "avg_clustering": nx.average_clustering(graph) if n_nodes else float("nan"),
    }


def _extended_network_summary(graph: nx.Graph) -> dict:
    """Extra global structural metrics: transitivity and giant-component ASP.

    Returns:
        Dict with keys: transitivity, giant_component_avg_shortest_path.
        NaN for giant_component_avg_shortest_path if the giant component has <2 nodes.
    """
    result: dict = {"transitivity": nx.transitivity(graph) or float("nan")}
    largest_cc = max(nx.connected_components(graph), key=len, default=set())
    if len(largest_cc) >= 2:
        subgraph = graph.subgraph(largest_cc)
        result["giant_component_avg_shortest_path"] = nx.average_shortest_path_length(subgraph)
    else:
        result["giant_component_avg_shortest_path"] = float("nan")
    return result


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


@dataclass
class EvolutionMetricsResult:
    """Aggregated outputs of :func:`compute_evolution_metrics`.

    Attributes:
        node_metrics: Long-format per-node metrics (window_end, node, metric, value).
        network_metrics: One row per window of network-level + extended structural metrics.
        graphs: window_end -> pruned, strength-annotated graph (for downstream chapters,
            e.g. community detection, that need the graph itself rather than a summary row).
    """

    node_metrics: pl.DataFrame
    network_metrics: pl.DataFrame
    graphs: dict[date, nx.Graph]


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
) -> EvolutionMetricsResult:
    """Compute node- and network-level metrics for every window.

    Args:
        df_returns: Long-format daily-returns dataframe.
        dates: Sorted unique trading dates.
        cfg: Evolution configuration.
        date_column, name_column, value_column: Column names.
        progress: Progress callback (done, total, desc).
        status: Status message callback.

    Returns:
        EvolutionMetricsResult with node-level metrics (long format), network-level
        metrics (one row per window), and the per-window graphs themselves.
    """
    if status is not None:
        status(f"Generating {cfg.window_size}-obs windows (step={cfg.step})...")

    windows = list(
        generate_windows(dates, cfg.window_size, cfg.step, expanding=cfg.expanding)
    )
    node_rows = []
    network_rows = []
    graphs: dict[date, nx.Graph] = {}
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
        network_row = _network_summary(graph, window_start, window_end)
        network_row.update(_extended_network_summary(graph))
        network_rows.append(network_row)
        graphs[window_end] = graph

    if progress is not None:
        progress(n_windows, n_windows, "done")

    return EvolutionMetricsResult(
        node_metrics=pl.DataFrame(node_rows),
        network_metrics=pl.DataFrame(network_rows),
        graphs=graphs,
    )


def build_adjacency_tensor(
    graphs: dict[date, nx.Graph],
    *,
    min_nodes: int = 2,
) -> tuple[list[date], list[str], np.ndarray]:
    """Stack per-window graphs into a binary (T, n, n) adjacency tensor.

    Restricts to the intersection of node sets across all collected windows
    (computed, not assumed) and uses binary edges rather than the continuous
    dissimilarity 'weight' attribute (matches graspologic's RDPG/ASE assumptions).

    Args:
        graphs: window_end -> nx.Graph, as returned in EvolutionMetricsResult.graphs.
        min_nodes: Minimum common-node count required (raises otherwise).

    Returns:
        (window_ends, common_nodes, tensor): window_ends is the sorted list of
        window-end dates (tensor axis-0 order); common_nodes is the sorted node
        names shared by every graph (tensor axis-1/2 order); tensor has shape
        (T, n, n), dtype float64, values in {0.0, 1.0}.

    Raises:
        ValueError: If fewer than min_nodes nodes are common to every window,
            or fewer than 2 windows are available.
    """
    window_ends = sorted(graphs.keys())
    if len(window_ends) < 2:
        raise ValueError("Need at least 2 windows to build an adjacency tensor.")
    node_sets = [set(graphs[we].nodes()) for we in window_ends]
    common_nodes = sorted(set.intersection(*node_sets))
    if len(common_nodes) < min_nodes:
        raise ValueError(
            f"Only {len(common_nodes)} common nodes across all windows -- "
            f"too few for community detection."
        )
    tensor = np.stack(
        [
            nx.to_numpy_array(graphs[we], nodelist=common_nodes, weight=None)
            for we in window_ends
        ]
    )
    return window_ends, common_nodes, tensor


def compute_window_communities(
    adjacency: np.ndarray,
    *,
    max_clusters: int = 10,
    ase_n_components: int | None = None,
    random_state: int = 0,
) -> np.ndarray:
    """ASE + graspologic KMeansCluster (auto-selects k via silhouette).

    Args:
        adjacency: Binary (n, n) adjacency matrix.
        max_clusters: Upper bound for k search (default 10).
        ase_n_components: Latent dimension for ASE (if None, uses sqrt(n)).
        random_state: Seed for reproducibility.

    Returns:
        labels: (n,) 0-indexed cluster assignments.
    """
    if ase_n_components is None:
        ase_n_components = max(2, int(np.sqrt(adjacency.shape[0])))
    # check_lcc=False: pruned correlation-network windows are routinely
    # disconnected by design (independent_threshold removes weak edges), so the
    # "not fully connected" warning is expected noise here, not a data problem.
    # It only gates the warning -- ASE still embeds the full adjacency either way.
    embedding = AdjacencySpectralEmbed(
        n_components=ase_n_components, check_lcc=False
    ).fit_transform(adjacency)
    kmeans = KMeansCluster(max_clusters=max_clusters, random_state=random_state)
    labels = kmeans.fit_predict(embedding)
    return labels


def compute_community_metrics(
    graphs: dict[date, nx.Graph],
    *,
    max_clusters: int = 10,
    min_nodes: int = 2,
    random_state: int = 0,
    progress: ProgressCallback | None = None,
    status: StatusCallback | None = None,
) -> pl.DataFrame:
    """Detect per-window communities (ASE+KMeans) across a common node set.

    Args:
        graphs: window_end -> nx.Graph, as returned in EvolutionMetricsResult.graphs.
        max_clusters: Upper bound for per-window k search.
        min_nodes: Minimum common-node count required.
        random_state: Seed for reproducibility.
        progress: Progress callback (done, total, desc).
        status: Status message callback.

    Returns:
        Long-format DataFrame: window_idx, window_end, node, community (int).

    Raises:
        ValueError: If too few common nodes or windows are available (propagated
            from build_adjacency_tensor).
    """
    if status is not None:
        status("Building common-node adjacency tensor...")
    window_ends, common_nodes, tensor = build_adjacency_tensor(graphs, min_nodes=min_nodes)

    n_windows = len(window_ends)
    rows = []
    for t in range(n_windows):
        if progress is not None:
            progress(t, n_windows, f"{window_ends[t]}")
        labels = compute_window_communities(
            tensor[t], max_clusters=max_clusters, random_state=random_state
        )
        for node_idx, node_name in enumerate(common_nodes):
            rows.append(
                {
                    "window_idx": t,
                    "window_end": window_ends[t],
                    "node": node_name,
                    "community": int(labels[node_idx]),
                }
            )

    if progress is not None:
        progress(n_windows, n_windows, "done")

    return pl.DataFrame(rows)


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
