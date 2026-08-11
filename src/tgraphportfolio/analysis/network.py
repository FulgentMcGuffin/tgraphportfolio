"""Wide-format pivoting and NetworkX construction from a connection matrix."""

from __future__ import annotations

import networkx as nx
import numpy as np
import polars as pl


def pivot_to_wide(
    df: pl.DataFrame,
    date_column: str,
    name_column: str,
    value_column: str,
) -> pl.DataFrame:
    """Pivot long data to wide format: dates as rows, nodes as columns."""
    pivoted = df.pivot(
        on=name_column, index=date_column, values=value_column
    ).sort(date_column)
    node_cols = sorted(col for col in pivoted.columns if col != date_column)
    return pivoted.select([date_column, *node_cols])


def build_corr_nx(
    measure_df: pl.DataFrame,
    independent_threshold: float = 0.33,
) -> nx.Graph:
    """Build a similarity network from a square connection-measure matrix.

    Measure values near 1 mean strong dependence. Edges are stored as
    ``1 - measure`` so strongly related nodes sit closer in force layouts.
    Weak edges (measure <= threshold) and self-loops are removed.
    """
    cor_matrix = measure_df.to_numpy().astype(float)
    sim_matrix = 1.0 - cor_matrix
    G = nx.from_numpy_array(sim_matrix)
    node_names = np.array(measure_df.columns)
    G = nx.relabel_nodes(G, lambda x: node_names[x])
    H = G.copy()
    for u, v, wt in G.edges.data("weight"):
        if wt >= 1.0 - independent_threshold:
            H.remove_edge(u, v)
        elif u == v:
            H.remove_edge(u, v)
    return H
