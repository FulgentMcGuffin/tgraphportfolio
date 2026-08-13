"""Pairwise connection measures between node time series."""

from __future__ import annotations

from collections.abc import Callable

import dcor
import numpy as np
import polars as pl
from scipy import stats

ProgressCallback = Callable[[int, int, str], None]


def distance_correlation_matrix(
    df_wide: pl.DataFrame,
    nodes: list[str],
    progress: ProgressCallback | None = None,
) -> pl.DataFrame:
    """Compute a square distance-correlation matrix for wide-format columns."""
    dcor_values = {node: {} for node in nodes}
    n_pairs = sum(len(nodes[k:]) for k in range(len(nodes)))
    done = 0
    k = 0
    for i in nodes:
        v_i = df_wide.get_column(i).to_numpy()
        for j in nodes[k:]:
            if progress is not None:
                progress(done, n_pairs, f"{i} / {j}")
            v_j = df_wide.get_column(j).to_numpy()
            vi_vj = np.column_stack((v_i, v_j))
            vi_vj = vi_vj[~np.isnan(vi_vj).any(axis=1)]
            if len(vi_vj) < 3:
                dcor_val = float("nan")
            else:
                dcor_val = float(
                    dcor.distance_correlation(vi_vj[:, 0], vi_vj[:, 1])
                )
            dcor_values[i][j] = dcor_val
            dcor_values[j][i] = dcor_val
            done += 1
        k += 1
    if progress is not None:
        progress(n_pairs, n_pairs, "done")
    return pl.DataFrame(
        {row: [dcor_values[row][col] for col in nodes] for row in nodes}
    )


def pearson_correlation_matrix(
    df_wide: pl.DataFrame,
    nodes: list[str],
    progress: ProgressCallback | None = None,
) -> pl.DataFrame:
    """Compute a square Pearson correlation matrix for wide-format columns.

    Handles NaN values and zero-variance columns gracefully.
    """
    pearson_values = {node: {} for node in nodes}
    n_pairs = sum(len(nodes[k:]) for k in range(len(nodes)))
    done = 0
    k = 0
    for i in nodes:
        v_i = df_wide.get_column(i).to_numpy()
        for j in nodes[k:]:
            if progress is not None:
                progress(done, n_pairs, f"{i} / {j}")
            v_j = df_wide.get_column(j).to_numpy()
            vi_vj = np.column_stack((v_i, v_j))
            vi_vj = vi_vj[~np.isnan(vi_vj).any(axis=1)]
            if len(vi_vj) < 3:
                pearson_val = float("nan")
            else:
                # Check for zero variance (correlation undefined)
                if np.std(vi_vj[:, 0]) == 0 or np.std(vi_vj[:, 1]) == 0:
                    pearson_val = float("nan")
                else:
                    corr = np.corrcoef(vi_vj[:, 0], vi_vj[:, 1])[0, 1]
                    # Ensure it's a scalar, not array, and handle NaN
                    pearson_val = float(corr) if not np.isnan(corr) else float("nan")
            pearson_values[i][j] = pearson_val
            pearson_values[j][i] = pearson_val
            done += 1
        k += 1
    if progress is not None:
        progress(n_pairs, n_pairs, "done")
    return pl.DataFrame(
        {row: [pearson_values[row][col] for col in nodes] for row in nodes}
    )


def spearman_correlation_matrix(
    df_wide: pl.DataFrame,
    nodes: list[str],
    progress: ProgressCallback | None = None,
) -> pl.DataFrame:
    """Compute a square Spearman correlation matrix for wide-format columns.

    Handles NaN values and constant sequences gracefully.
    """
    spearman_values = {node: {} for node in nodes}
    n_pairs = sum(len(nodes[k:]) for k in range(len(nodes)))
    done = 0
    k = 0
    for i in nodes:
        v_i = df_wide.get_column(i).to_numpy()
        for j in nodes[k:]:
            if progress is not None:
                progress(done, n_pairs, f"{i} / {j}")
            v_j = df_wide.get_column(j).to_numpy()
            vi_vj = np.column_stack((v_i, v_j))
            vi_vj = vi_vj[~np.isnan(vi_vj).any(axis=1)]
            if len(vi_vj) < 3:
                spearman_val = float("nan")
            else:
                try:
                    corr, _ = stats.spearmanr(vi_vj[:, 0], vi_vj[:, 1])
                    # Ensure it's a scalar and handle NaN
                    spearman_val = float(corr) if not np.isnan(corr) else float("nan")
                except (ValueError, RuntimeError):
                    # Handle edge cases (constant sequences, all NaN, etc.)
                    spearman_val = float("nan")
            spearman_values[i][j] = spearman_val
            spearman_values[j][i] = spearman_val
            done += 1
        k += 1
    if progress is not None:
        progress(n_pairs, n_pairs, "done")
    return pl.DataFrame(
        {row: [spearman_values[row][col] for col in nodes] for row in nodes}
    )


MEASURES: dict[str, Callable[..., pl.DataFrame]] = {
    "distance_correlation": distance_correlation_matrix,
    "pearson_correlation": pearson_correlation_matrix,
    "spearman_correlation": spearman_correlation_matrix,
}

MEASURE_LABELS: dict[str, str] = {
    "distance_correlation": "Distance correlation",
    "pearson_correlation": "Pearson correlation",
    "spearman_correlation": "Spearman correlation",
}


def available_measures() -> list[tuple[str, str]]:
    return [(key, MEASURE_LABELS.get(key, key)) for key in MEASURES]


def compute_measure(
    measure_id: str,
    df_wide: pl.DataFrame,
    nodes: list[str],
    progress: ProgressCallback | None = None,
) -> pl.DataFrame:
    fn = MEASURES.get(measure_id)
    if fn is None:
        raise ValueError(f"Unknown measure: {measure_id!r}")
    return fn(df_wide, nodes, progress=progress)
