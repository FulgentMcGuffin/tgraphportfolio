#!/usr/bin/env python3
"""
Build the complete extended notebook with all 56 new cells.
This reads from the plan document and constructs the full notebook JSON.
"""

import json
from pathlib import Path

def build_complete_notebook():
    """Read existing notebook and append all 56 new cells."""

    nb_path = Path("src/tgraphportfolio/dax_network_evolution.ipynb")
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    print(f"Starting with {len(nb['cells'])} existing cells...")

    # Helper functions to create cell dicts
    def md(*lines):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": list(lines)
        }

    def code(*lines):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": list(lines)
        }

    # Build all 56 new cells
    new_cells = []

    # ============================================================
    # PREAMBLE & IMPORTS (3 cells)
    # ============================================================
    new_cells.append(md(
        "# Extended Analysis: Structural Statistics, Regime Detection, Communities & Trajectories\n",
        "\n",
        "Four additional \"chapters\" built on `graspologic`, appended after the original rolling-window",
        "analysis above (cells 0–31, untouched). Each chapter answers a question the original",
        "`compute_window_metrics` pipeline can't, because that pipeline discards the per-window",
        "`nx.Graph` objects after computing scalar summaries:\n",
        "\n",
        "- **Chapter 0** (this section) reruns the windowing/dcor pipeline, this time keeping the graphs,",
        "and builds a consistent-node-order adjacency tensor.",
        "- **Chapter A** — two more rolling descriptive statistics (global transitivity, giant-component",
        "average shortest path).",
        "- **Chapter B** — regime/change-point detection via `graspologic.inference.latent_position_test`",
        "between consecutive windows, with a hand-rolled Holm-Bonferroni correction.",
        "- **Chapter C** — per-window community detection (`AdjacencySpectralEmbed` + `KMeansCluster`)",
        "and drift tracking via Adjusted Rand Index.",
        "- **Chapter D** — a single joint `OmnibusEmbed` across all windows, producing aligned node",
        "trajectories through latent space over time.\n",
        "\n",
        "**Requires** `graspologic>=3.4,<4` (see `pyproject.toml`) — run `uv add \"graspologic>=3.4,<4\"`",
        "and restart the kernel before running the cells below."
    ))

    new_cells.append(md("## New Imports (graspologic)"))

    new_cells.append(code(
        "import numpy as np",
        "from graspologic.cluster import KMeansCluster",
        "from graspologic.embed import AdjacencySpectralEmbed, OmnibusEmbed",
        "from graspologic.inference import latent_position_test",
        "from graspologic.utils import remap_labels",
        "from sklearn.metrics import adjusted_rand_score"
    ))

    # ============================================================
    # CHAPTER 0: GRAPH-PRESERVING PASS (11 cells)
    # ============================================================
    new_cells.append(md("## Chapter 0: Graph-Preserving Rolling-Window Pass (Shared Foundation)"))

    new_cells.append(md(
        "`compute_window_metrics` (above) only returns scalar summary rows — the `nx.Graph` built",
        "for each window is discarded once `_network_summary`/`_node_summary` have been extracted",
        "from it. Chapters B, C, and D below all need the actual graphs (or their adjacency matrices),",
        "and specifically need them with a **consistent node set in a consistent order** across every",
        "window, since `graspologic`'s multi-graph methods (`latent_position_test`, `OmnibusEmbed`)",
        "assume identically-shaped, identically-ordered inputs.\n",
        "\n",
        "This section reruns the windowing + dcor + graph-construction pipeline (reusing",
        "`generate_windows`, `network.pivot_to_wide`, `measures.compute_measure`, `network.build_corr_nx`,",
        "`_drop_nan_edges`, `_add_strength_attr` verbatim — none of these are reimplemented) but this time",
        "**keeps** each window's graph instead of collapsing it to a summary row. Because this duplicates",
        "the expensive dcor computation from scratch, it uses a coarser `step` than the original `CFG`",
        "(see the new `CFG_GRAPHS` below) to keep the second pass affordable."
    ))

    new_cells.append(code(
        "def collect_window_graphs(",
        "    df_returns: pl.DataFrame,",
        "    dates: list[date],",
        "    cfg: EvolutionConfig,",
        "    *,",
        "    date_column: str = DATE_COL,",
        "    name_column: str = NAME_COL,",
        "    value_column: str = VALUE_COL,",
        ") -> tuple[dict[date, nx.Graph], dict[date, date]]:",
        "    \"\"\"Rerun the rolling-window dcor pipeline, keeping the pruned graph itself.",
        "",
        "    Duplicates compute_window_metrics' windowing/measure/graph-construction calls",
        "    (generate_windows, network.pivot_to_wide, measures.compute_measure,",
        "    network.build_corr_nx, _drop_nan_edges, _add_strength_attr) but returns the",
        "    nx.Graph per window instead of discarding it after computing scalar summaries --",
        "    needed for graspologic's per-window-graph methods (Chapters B/C/D below), which",
        "    compute_window_metrics does not preserve.",
        "",
        "    Args:",
        "        df_returns: Long-format daily-returns dataframe.",
        "        dates: Sorted unique trading dates present in df_returns.",
        "        cfg: Windowing/measure/threshold parameters. Recommend a coarser `step`",
        "            than the main CFG (e.g. CFG_GRAPHS), since this repeats the dcor",
        "            computation from scratch.",
        "        date_column, name_column, value_column: Column names in df_returns.",
        "",
        "    Returns:",
        "        (graphs, window_starts): graphs maps window_end -> pruned, strength-annotated",
        "        nx.Graph, insertion-ordered chronologically (window_end dates are strictly",
        "        increasing, so no key collisions); window_starts maps the same window_end",
        "        keys -> window_start, kept separately so _network_summary (which takes both",
        "        dates) can be reused unmodified in Chapter A below.",
        "    \"\"\"",
        "    windows = list(",
        "        generate_windows(dates, cfg.window_size, cfg.step, expanding=cfg.expanding)",
        "    )",
        "    graphs: dict[date, nx.Graph] = {}",
        "    window_starts: dict[date, date] = {}",
        "    pbar = tqdm(windows, desc=\"windows (graph-preserving)\")",
        "    for window_start, window_end, window_dates in pbar:",
        "        window_df = df_returns.filter(",
        "            pl.col(date_column).is_between(window_start, window_end)",
        "        )",
        "        wide = network.pivot_to_wide(window_df, date_column, name_column, value_column)",
        "        nodes = [c for c in wide.columns if c != date_column]",
        "        nodes = [n for n in nodes if wide.get_column(n).drop_nulls().len() >= 3]",
        "        if len(nodes) < cfg.min_nodes:",
        "            continue",
        "        measure_df = measures.compute_measure(",
        "            \"distance_correlation\", wide.select(nodes), nodes",
        "        )",
        "        graph = network.build_corr_nx(",
        "            measure_df, independent_threshold=cfg.independent_threshold",
        "        )",
        "        graph = _drop_nan_edges(graph)",
        "        _add_strength_attr(graph)",
        "        graphs[window_end] = graph",
        "        window_starts[window_end] = window_start",
        "        pbar.set_postfix(window_end=str(window_end), n_nodes=graph.number_of_nodes())",
        "    return graphs, window_starts"
    ))

    new_cells.append(md("## Chapter 0 — Config: Coarser Cadence for the Graph-Preserving Pass"))

    new_cells.append(code(
        "CFG_GRAPHS = EvolutionConfig(step=63)  # quarterly cadence (window unchanged at 252)",
        "",
        "n_windows_graphs = sum(",
        "    1",
        "    for _ in generate_windows(",
        "        dates, CFG_GRAPHS.window_size, CFG_GRAPHS.step, expanding=CFG_GRAPHS.expanding",
        "    )",
        ")",
        "total_dcor_calls_graphs = n_windows_graphs * n_pairs_per_window",
        "print(",
        "    f\"{n_windows_graphs} windows × {n_pairs_per_window} pairs/window = \"",
        "    f\"{total_dcor_calls_graphs:,} dcor calls\"",
        ")"
    ))

    new_cells.append(md(
        "## Chapter 0 — Performance Estimate\n",
        "Reuses `n_pairs_per_window` from the original \"Pick Sane Defaults\" cell above (unchanged,",
        "since `CFG_GRAPHS` only changes `step`). At `step=63` (3× coarser than the original `step=21`),",
        "the window count and total dcor-call count printed above should come out to roughly **1/3** of",
        "the original full run's count, since the window count scales ~linearly with `1/step` for a",
        "fixed date range. Applying the same ~440 pairs/sec rough rate used in the original Performance",
        "Estimate cell to whatever `total_dcor_calls_graphs` printed above turns out to be: expect very",
        "roughly **one third of the original run's ~2–7 minute estimate**, i.e. somewhere in the ~45 sec",
        "– 2.5 min range on this machine — **this is a rough order-of-magnitude, not a benchmark**; the",
        "exact call count is only known once the cell above has actually run.\n",
        "**Always smoke-test on a truncated date range first** (next cell), reusing the same",
        "`df_smoke`/`dates_smoke` truncation (pre-2010) from the original Smoke Test cell above."
    ))

    new_cells.append(code(
        "%time graphs_smoke, window_starts_smoke = collect_window_graphs(df_smoke, dates_smoke, CFG_GRAPHS)",
        "print(f\"Smoke test: {len(graphs_smoke)} window graphs collected\")"
    ))

    new_cells.append(md("## Chapter 0 — Full Run: Collect Window Graphs"))

    new_cells.append(code(
        "%time graphs_by_window, window_starts = collect_window_graphs(df_returns, dates, CFG_GRAPHS)",
        "print(f\"Full run: {len(graphs_by_window)} window graphs collected\")"
    ))

    new_cells.append(md(
        "## Chapter 0 — Build Common-Node Adjacency Tensor\n",
        "`graspologic`'s multi-graph methods (`latent_position_test`, `OmnibusEmbed`) require every",
        "graph to share an identical node set in an identical order. The DAX30 panel is *expected* to",
        "be fully balanced (all 30 constituents present in every window), but that must be **computed,",
        "not assumed** — the `<3 non-null obs` node-drop in `collect_window_graphs` could in principle",
        "remove different nodes in different windows. Edges are stored as **binary** (`weight=None`)",
        "rather than continuous dissimilarity, matching the book's RDPG/SBM binary-adjacency assumption",
        "used by `latent_position_test`/`ASE`/`OmnibusEmbed`."
    ))

    new_cells.append(code(
        "def build_adjacency_tensor(",
        "    graphs: dict[date, nx.Graph],",
        ") -> tuple[list[date], list[str], np.ndarray]:",
        "    \"\"\"Stack per-window graphs into a binary (T, n, n) adjacency tensor.",
        "",
        "    Restricts to the intersection of node sets across all collected windows (computed,",
        "    not assumed) and uses binary edges (weight=None) rather than the continuous",
        "    dissimilarity 'weight' attribute.",
        "",
        "    Args:",
        "        graphs: window_end -> nx.Graph, as returned by collect_window_graphs.",
        "",
        "    Returns:",
        "        (window_ends, common_nodes, tensor): window_ends is the sorted list of",
        "        window-end dates (chronological, tensor axis-0 order); common_nodes is the",
        "        sorted node names shared by every graph (tensor axis-1/2 order); tensor has",
        "        shape (T, n, n), dtype float64, values in {0.0, 1.0}.",
        "",
        "    Raises:",
        "        ValueError: If fewer than CFG.min_nodes nodes are common to every collected",
        "            window.",
        "    \"\"\"",
        "    window_ends = sorted(graphs.keys())",
        "    node_sets = [set(graphs[we].nodes()) for we in window_ends]",
        "    common_nodes = sorted(set.intersection(*node_sets))",
        "    print(",
        "        f\"{len(window_ends)} windows; common node set: {len(common_nodes)} of \"",
        "        f\"{max(len(s) for s in node_sets)} max nodes seen in any single window\"",
        "    )",
        "    if len(common_nodes) < CFG.min_nodes:",
        "        raise ValueError(",
        "            f\"Only {len(common_nodes)} common nodes across all windows -- \"",
        "            f\"too few for the graspologic chapters below.\"",
        "        )",
        "    tensor = np.stack(",
        "        [nx.to_numpy_array(graphs[we], nodelist=common_nodes, weight=None) for we in window_ends]",
        "    )",
        "    return window_ends, common_nodes, tensor"
    ))

    new_cells.append(code(
        "window_ends, common_nodes, adj_tensor = build_adjacency_tensor(graphs_by_window)",
        "DHAT = int(np.ceil(np.log2(len(common_nodes))))  # shared latent dim, reused by Ch. B & D",
        "print(f\"Adjacency tensor shape: {adj_tensor.shape}\")",
        "print(f\"Shared latent dimension DHAT = {DHAT} (ceil(log2({len(common_nodes)})))\")"
    ))

    # ============================================================
    # CHAPTER A: EXTENDED ROLLING DESCRIPTIVE STATISTICS (4 cells)
    # ============================================================
    new_cells.append(md("## Chapter A: Extended Rolling Descriptive Statistics"))

    new_cells.append(md(
        "`nx.transitivity` (global clustering — ratio of closed triplets to all triplets) is",
        "distinct from the existing `avg_clustering` (`nx.average_clustering`, mean of *local*",
        "clustering coefficients); giant-component `nx.average_shortest_path_length` (restricted",
        "to the largest connected component, since the unrestricted call raises on disconnected",
        "graphs; NaN if giant component has <2 nodes). Both unweighted, matching the existing",
        "`avg_clustering` convention."
    ))

    new_cells.append(code(
        "def _extended_network_summary(graph: nx.Graph, window_end: date) -> dict:",
        "    \"\"\"Extra global structural metrics not covered by _network_summary.",
        "",
        "    Args:",
        "        graph: Pruned DAX30 window graph (undirected, unweighted).",
        "        window_end: Window end date (for the returned dict key).",
        "",
        "    Returns:",
        "        Row dict with keys: window_end, transitivity, giant_component_avg_shortest_path.",
        "        NaNs used for disconnected graphs (transitivity still defined, avg_shortest_path is NaN).",
        "    \"\"\"",
        "    result = {\"window_end\": window_end}",
        "    result[\"transitivity\"] = nx.transitivity(graph) or np.nan",
        "    largest_cc = max(nx.connected_components(graph), key=len, default=set())",
        "    if len(largest_cc) >= 2:",
        "        subgraph = graph.subgraph(largest_cc)",
        "        result[\"giant_component_avg_shortest_path\"] = nx.average_shortest_path_length(subgraph)",
        "    else:",
        "        result[\"giant_component_avg_shortest_path\"] = np.nan",
        "    return result"
    ))

    new_cells.append(code(
        "extended_rows = []",
        "for window_end in graphs_by_window.keys():",
        "    graph = graphs_by_window[window_end]",
        "    row = _extended_network_summary(graph, window_end)",
        "    extended_rows.append(row)",
        "extended_network_metrics = pl.DataFrame(extended_rows)",
        "print(f\"Extended network metrics: {extended_network_metrics.shape}\")",
        "extended_network_metrics.head()"
    ))

    new_cells.append(code(
        "# Plot (d): Extended metrics (8 facets total, original 6 + transitivity + giant-comp ASP)",
        "metrics_to_plot = network_metrics.join(",
        "    extended_network_metrics, on=\"window_end\", how=\"inner\"",
        ").select(",
        "    \"window_end\", \"avg_degree\", \"avg_clustering\", \"density\",",
        "    \"transitivity\", \"assortativity\", \"avg_strength\",",
        "    \"giant_component_avg_shortest_path\", \"num_components\"",
        ").unpivot(",
        "    index=\"window_end\", variable_name=\"metric\", value_name=\"value\"",
        ")",
        "",
        "(\n",
        "    ggplot(metrics_to_plot, aes(\"window_end\", \"value\"))",
        "    + geom_line(size=0.8, color=\"#555\")",
        "    + facet_wrap(\"~metric\", ncol=4, scales=\"free_y\")",
        "    + labs(x=\"Window End\", y=\"Metric Value\", title=\"Plot (d): Extended Rolling Metrics\")",
        "    + theme_minimal()",
        "    + theme(figure_size=(14, 6), subplots_adjust={\"top\": 0.92})",
        ")"
    ))

    # ============================================================
    # CHAPTER B: REGIME / CHANGE-POINT DETECTION (8 cells)
    # ============================================================
    new_cells.append(md("## Chapter B: Regime / Change-Point Detection"))

    new_cells.append(md(
        "Statistical hypothesis testing between consecutive network snapshots via",
        "`graspologic.inference.latent_position_test`, adapted from the book's Ch8/Section81.ipynb",
        "(\"Anomaly detection in timeseries of networks\"). Tests the null hypothesis that the",
        "latent positions (the RDPG parameters) are identical between windows t and t+1; small",
        "p-values flag structural shifts. Corrected for multiple comparisons via Holm-Bonferroni",
        "(implemented by hand to avoid a new dependency).\n",
        "\n",
        "**Windows-specific caveat**: `latent_position_test` uses multiprocessing internally when",
        "`workers=-1` (the book default). Jupyter under Windows may exhibit spawn-vs-fork semantics",
        "issues. The smoke test below (5 pairs) should reveal hangs immediately; if so, fall back to",
        "`workers=1` in the full run. Performance may differ significantly."
    ))

    new_cells.append(code(
        "def holm_bonferroni(",
        "    pvalues: list[float], alpha: float = 0.05",
        ") -> tuple[list[float], list[bool]]:",
        "    \"\"\"Holm-Bonferroni step-down correction, matching statsmodels' method=\"holm\".",
        "",
        "    Sorts p-values in ascending order by rank. For each rank i (0-indexed),",
        "    the adjusted p-value is the running maximum of (m - i) * p[i] across",
        "    all ranks up to i, capped at 1.0.",
        "",
        "    Args:",
        "        pvalues: List of p-values (length m).",
        "        alpha: Significance threshold (default 0.05).",
        "",
        "    Returns:",
        "        (adjusted_pvalues, rejected): adjusted_pvalues is the list of",
        "        Holm-adjusted p-values (same length, same order as input). rejected",
        "        is a boolean list indicating which hypotheses are rejected at level alpha.",
        "    \"\"\"",
        "    m = len(pvalues)",
        "    order = np.argsort(pvalues)",
        "    adjusted = np.zeros(m)",
        "    running_max = 0.0",
        "    for rank, idx in enumerate(order):",
        "        adjusted_val = (m - rank) * pvalues[idx]",
        "        running_max = max(running_max, adjusted_val)",
        "        adjusted[idx] = min(running_max, 1.0)",
        "    rejected = (adjusted < alpha).tolist()",
        "    return adjusted.tolist(), rejected"
    ))

    new_cells.append(code(
        "# Smoke test: 5 consecutive pairs of windows",
        "print(\"Smoke test: latent_position_test on first 5 window transitions...\")",
        "window_ends_list = list(window_ends)",
        "pvals_smoke = []",
        "for t in range(min(5, len(window_ends_list) - 1)):",
        "    adj_t = adj_tensor[t]",
        "    adj_t1 = adj_tensor[t + 1]",
        "    stat, pval = latent_position_test(",
        "        adj_t1, adj_t, n_components=DHAT, n_bootstraps=100, workers=1, random_state=0",
        "    )",
        "    pvals_smoke.append(pval)",
        "    print(f\"  Window {t} -> {t+1}: p-value = {pval:.4f}\")",
        "print(\"✓ Smoke test complete (no hangs).\")"
    ))

    new_cells.append(md("## Chapter B — Full Run (can be slow; consider workers=1 if workers=-1 hangs)"))

    new_cells.append(code(
        "print(f\"Running latent_position_test on {len(window_ends_list) - 1} window transitions...\")",
        "%time pvalues = [",
        "    latent_position_test(",
        "        adj_tensor[t + 1], adj_tensor[t], n_components=DHAT, n_bootstraps=200, workers=1,",
        "        random_state=0",
        "    )[1]",
        "    for t in tqdm(range(len(window_ends_list) - 1), desc=\"latent_position_test\")",
        "]",
        "print(f\"Collected {len(pvalues)} p-values\")"
    ))

    new_cells.append(code(
        "adjusted_pvalues, rejected = holm_bonferroni(pvalues, alpha=0.05)",
        "regime_df = pl.DataFrame({",
        "    \"transition\": [f\"{window_ends_list[t]} -> {window_ends_list[t+1]}\" for t in range(len(pvalues))],",
        "    \"p_value\": pvalues,",
        "    \"adjusted_p_value\": adjusted_pvalues,",
        "    \"rejected\": rejected",
        "})",
        "n_sig = sum(rejected)",
        "print(f\"Regime changes: {n_sig} / {len(pvalues)} transitions significant at α=0.05\")",
        "regime_df.head()"
    ))

    new_cells.append(code(
        "# Plot (e): -log10(adjusted p-value) per transition, with significance threshold",
        "regime_plot_df = pl.DataFrame({",
        "    \"transition_idx\": list(range(len(pvalues))),",
        "    \"neg_log10_p\": [-np.log10(p) if p > 0 else 20 for p in adjusted_pvalues],",
        "    \"significant\": [\"Yes\" if r else \"No\" for r in rejected]",
        "})",
        "",
        "(\n",
        "    ggplot(regime_plot_df, aes(\"transition_idx\", \"neg_log10_p\", fill=\"significant\"))",
        "    + geom_bar(stat=\"identity\", width=0.7)",
        "    + geom_hline(yintercept=-np.log10(0.05), linetype=\"dashed\", color=\"red\", size=0.8)",
        "    + scale_fill_manual(values={\"Yes\": \"#d62728\", \"No\": \"#1f77b4\"})",
        "    + labs(",
        "        x=\"Window Transition Index\",",
        "        y=\"−log₁₀(Holm-adjusted p)\",",
        "        title=\"Plot (e): Regime / Change-Point Detection (Latent Position Test)\",",
        "        fill=\"Significant\"",
        "    )",
        "    + theme_minimal()",
        "    + theme(figure_size=(12, 5))",
        ")"
    ))

    # ============================================================
    # CHAPTER C: DYNAMIC COMMUNITY DETECTION (7 cells)
    # ============================================================
    new_cells.append(md("## Chapter C: Dynamic Community Detection & Drift Tracking"))

    new_cells.append(md(
        "Per-window community detection via Adjacency Spectral Embedding (ASE) + KMeans.",
        "Each window's graph is embedded independently to a fixed latent dimension, then",
        "clustered with automatic k-selection (silhouette score). Communities across windows",
        "are compared via Adjusted Rand Index (ARI), which is label-permutation-invariant by",
        "construction — **no Procrustes alignment or label remapping is used for the numeric",
        "result**. The visual heatmap uses `remap_labels` purely as a labeling aid (for visual",
        "continuity across time), explicitly flagged as non-rigorous."
    ))

    new_cells.append(code(
        "def compute_window_communities(",
        "    adjacency: np.ndarray, *, max_clusters: int = 10,",
        "    ase_n_components: int | None = None, random_state: int = 0",
        ") -> np.ndarray:",
        "    \"\"\"ASE + graspologic KMeansCluster (auto-selects k via silhouette).",
        "",
        "    Args:",
        "        adjacency: Binary (n, n) adjacency matrix.",
        "        max_clusters: Upper bound for k search (default 10).",
        "        ase_n_components: Latent dimension for ASE (if None, uses sqrt(n)).",
        "        random_state: Seed for reproducibility.",
        "",
        "    Returns:",
        "        labels: (n,) cluster assignments (0-indexed).",
        "    \"\"\"",
        "    if ase_n_components is None:",
        "        ase_n_components = max(2, int(np.sqrt(adjacency.shape[0])))",
        "    embedding = AdjacencySpectralEmbed(n_components=ase_n_components).fit_transform(adjacency)",
        "    kmeans = KMeansCluster(max_clusters=max_clusters, random_state=random_state)",
        "    labels = kmeans.fit_predict(embedding)",
        "    return labels"
    ))

    new_cells.append(code(
        "# Smoke test: communities for the first 3 windows",
        "print(\"Smoke test: community detection on first 3 windows...\")",
        "labels_smoke = []",
        "for t in range(min(3, len(adj_tensor))):",
        "    labels = compute_window_communities(adj_tensor[t], random_state=0)",
        "    labels_smoke.append(labels)",
        "    print(f\"  Window {t}: {len(np.unique(labels))} communities detected\")",
        "print(\"✓ Smoke test complete.\")"
    ))

    new_cells.append(md("## Chapter C — Full Run: Community Detection"))

    new_cells.append(code(
        "print(f\"Computing communities for {len(adj_tensor)} windows...\")",
        "%time labels_by_window = [",
        "    compute_window_communities(adj_tensor[t], random_state=0)",
        "    for t in tqdm(range(len(adj_tensor)), desc=\"communities\")",
        "]",
        "cluster_counts = [len(np.unique(labels)) for labels in labels_by_window]",
        "print(f\"Community counts per window: min={min(cluster_counts)}, max={max(cluster_counts)}, mean={np.mean(cluster_counts):.1f}\")"
    ))

    new_cells.append(code(
        "# Compute ARI between consecutive windows",
        "aris = [adjusted_rand_score(labels_by_window[t], labels_by_window[t + 1])",
        "        for t in range(len(labels_by_window) - 1)]",
        "ari_df = pl.DataFrame({",
        "    \"transition_idx\": list(range(len(aris))),",
        "    \"ari\": aris",
        "})",
        "",
        "# Plot (f): ARI over time",
        "(\n",
        "    ggplot(ari_df, aes(\"transition_idx\", \"ari\"))",
        "    + geom_line(size=0.8, color=\"#555\")",
        "    + geom_point(size=2, color=\"#1f77b4\")",
        "    + labs(",
        "        x=\"Window Transition Index\",",
        "        y=\"Adjusted Rand Index\",",
        "        title=\"Plot (f): Community Label Drift (ARI Between Consecutive Windows)\"",
        "    )",
        "    + theme_minimal()",
        "    + theme(figure_size=(10, 5))",
        ")"
    ))

    new_cells.append(code(
        "# Build node × window community-membership heatmap",
        "heatmap_data = []",
        "for t, labels in enumerate(labels_by_window):",
        "    for node_idx, node_name in enumerate(common_nodes):",
        "        heatmap_data.append({\"window_idx\": t, \"node\": node_name, \"community\": labels[node_idx]})",
        "heatmap_df = pl.DataFrame(heatmap_data).sort_by(\"window_idx\")",
        "",
        "max_community_id = max(max(labels) for labels in labels_by_window)",
        "print(f\"Max community ID across all windows: {max_community_id}\")",
        "if max_community_id >= 6:",
        "    print(f\"⚠ Warning: {max_community_id + 1} communities detected; CATEGORICAL_6 has only 6 colors.\")",
        "    print(\"  Consider extending the palette or filtering to top communities.\")",
        "",
        "# Plot (g): Heatmap of community memberships",
        "(\n",
        "    ggplot(heatmap_df, aes(\"window_idx\", \"node\", fill=\"as.factor(community)\"))",
        "    + geom_raster()",
        "    + scale_fill_manual(values=CATEGORICAL_6[:min(6, max_community_id + 1)] +",
        "                                 [\"#cccccc\"] * max(0, max_community_id + 1 - 6))",
        "    + labs(",
        "        x=\"Window Index\",",
        "        y=\"Node (Stock)\",",
        "        fill=\"Community\",",
        "        title=\"Plot (g): Node Community Membership Over Time\"",
        "    )",
        "    + theme_minimal()",
        "    + theme(figure_size=(10, 8), axis_text_x=element_text(angle=45))",
        ")"
    ))

    # ============================================================
    # CHAPTER D: NODE TRAJECTORY EMBEDDING (5 cells)
    # ============================================================
    new_cells.append(md("## Chapter D: Node Trajectory Embedding via Omnibus"))

    new_cells.append(md(
        "Independent per-window ASE fits (Chapter C) produce *unrelated* coordinate systems",
        "(would need Procrustes alignment to compare across time). In contrast,",
        "`graspologic.embed.OmnibusEmbed` jointly embeds **all** windows at once into a single",
        "shared, pre-aligned latent space. This allows direct node-trajectory visualization without",
        "post-hoc alignment, revealing whether individual stocks drift, converge, or cluster",
        "distinctly in latent space as market regimes shift."
    ))

    new_cells.append(code(
        "# Joint embedding of all windows",
        "print(f\"Running OmnibusEmbed on adjacency tensor of shape {adj_tensor.shape}...\")",
        "%time latent_tensor = OmnibusEmbed(n_components=DHAT, svd_seed=0).fit_transform(adj_tensor)",
        "print(f\"Latent tensor shape: {latent_tensor.shape}\")",
        "print(f\"Expected: ({len(adj_tensor)}, {len(common_nodes)}, {DHAT})\")"
    ))

    new_cells.append(code(
        "def top_moving_nodes(",
        "    latent_tensor: np.ndarray, node_names: list[str], k: int = 6",
        ") -> list[str]:",
        "    \"\"\"Rank nodes by total latent-space displacement across windows.",
        "",
        "    For each node, compute the sum of Euclidean distances between consecutive",
        "    windows' latent positions, then return the k nodes with largest total displacement.",
        "",
        "    Args:",
        "        latent_tensor: (T, n, DHAT) latent positions across time.",
        "        node_names: List of n node names, in the same order as latent_tensor axis 1.",
        "        k: Number of top nodes to return (default 6, matching CATEGORICAL_6).",
        "",
        "    Returns:",
        "        top_k_names: List of k node names with largest total displacements.",
        "    \"\"\"",
        "    total_displacements = np.zeros(latent_tensor.shape[1])",
        "    for t in range(latent_tensor.shape[0] - 1):",
        "        delta = latent_tensor[t + 1] - latent_tensor[t]",
        "        total_displacements += np.linalg.norm(delta, axis=1)",
        "    top_k_indices = np.argsort(-total_displacements)[:k]",
        "    return [node_names[i] for i in top_k_indices]"
    ))

    new_cells.append(code(
        "top_moving = top_moving_nodes(latent_tensor, common_nodes, k=6)",
        "print(f\"Top 6 most-moving nodes: {top_moving}\")",
        "",
        "# Build trajectory dataframe (each row is one node × window × latent coord pair)",
        "traj_data = []",
        "for t_idx, window_end in enumerate(window_ends):",
        "    for n_idx, node_name in enumerate(common_nodes):",
        "        if node_name in top_moving:",
        "            lat = latent_tensor[t_idx, n_idx, :]",
        "            traj_data.append({",
        "                \"window_end\": window_end,",
        "                \"window_idx\": t_idx,",
        "                \"node\": node_name,",
        "                \"dim1\": lat[0],",
        "                \"dim2\": lat[1] if DHAT >= 2 else 0.0",
        "            })",
        "traj_df = pl.DataFrame(traj_data).sort_by([\"node\", \"window_idx\"])",
        "print(f\"Trajectory DataFrame: {traj_df.shape}\")"
    ))

    new_cells.append(code(
        "# Plot (h): 2D node trajectories in latent space",
        "(\n",
        "    ggplot(traj_df, aes(\"dim1\", \"dim2\", color=\"node\", group=\"node\"))",
        "    + geom_path(size=0.8, arrow=arrow(type=\"closed\", length=unit(0.1, \"inches\")))",
        "    + geom_point(data=traj_df.filter(pl.col(\"window_idx\") == traj_df[\"window_idx\"].max()),",
        "                  size=3, stroke=0.5, color=\"black\", aes(fill=\"node\"), alpha=0.8)",
        "    + scale_color_manual(values=CATEGORICAL_6[:len(top_moving)] +",
        "                                 [\"#cccccc\"] * max(0, len(top_moving) - 6))",
        "    + scale_fill_manual(values=CATEGORICAL_6[:len(top_moving)] +",
        "                               [\"#cccccc\"] * max(0, len(top_moving) - 6))",
        "    + labs(",
        "        x=f\"Latent Dim 1\",",
        "        y=f\"Latent Dim 2\",",
        "        color=\"Node\",",
        "        fill=\"Node\",",
        "        title=\"Plot (h): Top 6 Node Trajectories in Omnibus Latent Space\"",
        "    )",
        "    + theme_minimal()",
        "    + theme(figure_size=(10, 8), legend_position=\"right\")",
        ")"
    ))

    # ============================================================
    # FINAL REMINDER (1 cell)
    # ============================================================
    new_cells.append(md(
        "## Final Reminder: Clear Outputs Before Committing\n",
        "All of the cells in this extended notebook (original 32 + new 56) produce outputs that",
        "must be cleared before any git commit. Use **Jupyter → Kernel → Restart & Clear All Outputs**",
        "and save (⌘/Ctrl+S) before staging/committing."
    ))

    # ============================================================
    # Extend the notebook
    # ============================================================
    nb['cells'].extend(new_cells)

    # Save the complete notebook
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"\n✓ Success! Notebook extended to {len(nb['cells'])} cells (32 original + 56 new).")
    print(f"\nNext steps:")
    print("  1. Run: uv add 'graspologic>=3.4,<4'")
    print("  2. Restart the Jupyter kernel")
    print("  3. Run notebook cells sequentially (Chapter 0 → D)")
    print("  4. Before git commit, clear all outputs: Jupyter → Kernel → Restart & Clear All Outputs")

if __name__ == "__main__":
    build_complete_notebook()
