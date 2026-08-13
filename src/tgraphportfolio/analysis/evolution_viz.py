"""Visualization for temporal network evolution."""

from __future__ import annotations

import io
from datetime import date

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import polars as pl


def render_weighted_degree_heatmap(
    node_metrics: pl.DataFrame,
    *,
    width: int = 12,
    height: int = 8,
    dpi: int = 100,
) -> bytes:
    """Render weighted-degree heatmap (Plot b) as PNG bytes.

    Args:
        node_metrics: Long-format DataFrame with columns
            (window_end, node, metric, value).
        width, height: Figure dimensions in inches.
        dpi: Dots per inch.

    Returns:
        PNG image bytes.
    """
    # Filter to weighted_degree metric
    heatmap_df = node_metrics.filter(pl.col("metric") == "weighted_degree")

    if heatmap_df.is_empty():
        return _empty_plot("No weighted-degree data available")

    # Pivot to matrix form
    pivot = heatmap_df.pivot(
        on="window_end", index="node", values="value", aggregate_function="first"
    )
    pivot = pivot.sort("node")  # Sort nodes for consistent axis

    nodes = pivot.get_column("node").to_list()
    window_ends = [col for col in pivot.columns if col != "node"]
    window_ends_sorted = sorted(window_ends)

    # Build matrix
    matrix = np.array(
        [pivot.filter(pl.col("node") == n).select(window_ends_sorted).to_numpy()[0]
         for n in nodes]
    )

    # Render
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    # Heatmap
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", interpolation="nearest")

    # Axes
    ax.set_xticks(np.arange(len(window_ends_sorted)))
    ax.set_yticks(np.arange(len(nodes)))
    ax.set_xticklabels([str(d) for d in window_ends_sorted], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(nodes, fontsize=6)

    ax.set_xlabel("Window end date", color="#e2e8f0", fontsize=9)
    ax.set_ylabel("Stock", color="#e2e8f0", fontsize=9)
    ax.set_title("DAX30 node weighted-degree evolution", color="#e2e8f0", fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Weighted degree", rotation=270, labelpad=20, color="#e2e8f0", fontsize=8)
    cbar.ax.tick_params(colors="#e2e8f0", labelsize=7)

    # Tick colors
    ax.tick_params(colors="#e2e8f0", labelsize=7)

    plt.tight_layout()

    # Export as PNG
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor="#0f172a", dpi=dpi)
    plt.close(fig)

    buf.seek(0)
    return buf.read()


def render_centrality_trajectories(
    node_metrics: pl.DataFrame,
    centrality_metric: str = "eigenvector",
    n_nodes: int = 10,
    *,
    width: int = 12,
    height: int = 8,
    dpi: int = 100,
) -> bytes:
    """Render centrality trajectories (Plot c) as PNG bytes.

    Args:
        node_metrics: Long-format DataFrame with columns
            (window_end, node, metric, value).
        centrality_metric: Which centrality to plot (default "eigenvector").
        n_nodes: Number of top variable nodes to plot (limited to 10).
        width, height: Figure dimensions in inches.
        dpi: Dots per inch.

    Returns:
        PNG image bytes.
    """
    # Filter to centrality metric
    cent_df = node_metrics.filter(pl.col("metric") == centrality_metric)

    if cent_df.is_empty():
        return _empty_plot(f"No {centrality_metric} centrality data available")

    # Find top variable nodes
    top_k = min(n_nodes, 10)
    top_nodes = (
        cent_df.group_by("node")
        .agg(pl.col("value").std().alias("std"))
        .sort("std", descending=True)
        .head(top_k)
        .get_column("node")
        .to_list()
    )

    line_df = cent_df.filter(pl.col("node").is_in(top_nodes))

    if line_df.is_empty():
        return _empty_plot("No data for top nodes")

    # Render
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    # Color palette (slightly muted for less prominence)
    colors = [
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd"
    ]

    # Line styles and markers for visual distinction
    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]
    markers = ["o", "s", "^", "D", "v", "p", "*", "h", "+", "x"]

    # Sort by window_end for consistent plotting
    pandas_df = line_df.to_pandas().sort_values("window_end")

    # Plot lines
    for i, node in enumerate(top_nodes):
        node_data = pandas_df[pandas_df["node"] == node]
        ax.plot(
            node_data["window_end"],
            node_data["value"],
            label=node,
            color=colors[i % len(colors)],
            linewidth=1.2,  # Thinner lines for less prominence
            linestyle=line_styles[i % len(line_styles)],  # Vary line style
            marker=markers[i % len(markers)],  # Vary marker shape
            markersize=3,  # Smaller markers
            alpha=0.8,  # Slightly transparent
        )

    ax.set_xlabel("Window end date", color="#e2e8f0", fontsize=9)
    ax.set_ylabel(f"{centrality_metric.title()} centrality", color="#e2e8f0", fontsize=9)
    ax.set_title(f"Most variable DAX30 nodes ({centrality_metric})", color="#e2e8f0", fontsize=12)

    ax.legend(loc="best", fontsize=7, facecolor="#0f172a", edgecolor="#e2e8f0",
              labelcolor="#e2e8f0", framealpha=0.9)
    ax.grid(True, alpha=0.2, color="#e2e8f0")
    ax.tick_params(colors="#e2e8f0", labelsize=7)

    # Axis spines
    for spine in ax.spines.values():
        spine.set_edgecolor("#e2e8f0")

    plt.tight_layout()

    # Export as PNG
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor="#0f172a", dpi=dpi)
    plt.close(fig)

    buf.seek(0)
    return buf.read()


def _empty_plot(message: str, width: int = 12, height: int = 8, dpi: int = 100) -> bytes:
    """Render a placeholder plot with a message."""
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center", fontsize=14,
        color="#94a3b8", transform=ax.transAxes
    )
    ax.set_xticks([])
    ax.set_yticks([])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor="#0f172a", dpi=dpi)
    plt.close(fig)

    buf.seek(0)
    return buf.read()
