#!/usr/bin/env python3
"""Build the extended dax_network_evolution notebook with 4 chapters from graspologic."""

import json
from pathlib import Path

# Read the existing notebook
nb_path = Path("src/tgraphportfolio/dax_network_evolution.ipynb")
with open(nb_path, 'r') as f:
    nb = json.load(f)

print(f"Existing notebook: {len(nb['cells'])} cells")

# All new cells (32-87, 56 total)
new_cells = [
    # ========== PREAMBLE ==========
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Extended Analysis: Structural Statistics, Regime Detection, Communities & Trajectories\n",
            "\n",
            "Four additional \"chapters\" built on `graspologic`, appended after the original\n",
            "rolling-window analysis above (cells 0–31, untouched). Each chapter answers a\n",
            "question the original `compute_window_metrics` pipeline can't, because that\n",
            "pipeline discards the per-window `nx.Graph` objects after computing scalar\n",
            "summaries:\n",
            "\n",
            "- **Chapter 0** (this section) reruns the windowing/dcor pipeline, this time\n",
            "keeping the graphs, and builds a consistent-node-order adjacency tensor.\n",
            "- **Chapter A** — two more rolling descriptive statistics (global transitivity,\n",
            "giant-component average shortest path).\n",
            "- **Chapter B** — regime/change-point detection via `graspologic.inference.latent_position_test`\n",
            "between consecutive windows, with a hand-rolled Holm-Bonferroni correction.\n",
            "- **Chapter C** — per-window community detection (`AdjacencySpectralEmbed` +\n",
            "`KMeansCluster`) and drift tracking via Adjusted Rand Index.\n",
            "- **Chapter D** — a single joint `OmnibusEmbed` across all windows, producing\n",
            "aligned node trajectories through latent space over time.\n",
            "\n",
            "**Requires** `graspologic>=3.4,<4` (see `pyproject.toml`) — run `uv add\n",
            "\"graspologic>=3.4,<4\"` and restart the kernel before running the cells below."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## New Imports (graspologic)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "from graspologic.cluster import KMeansCluster\n",
            "from graspologic.embed import AdjacencySpectralEmbed, OmnibusEmbed\n",
            "from graspologic.inference import latent_position_test\n",
            "from graspologic.utils import remap_labels\n",
            "from sklearn.metrics import adjusted_rand_score"
        ]
    },
    # ========== CHAPTER 0 ==========
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Chapter 0: Graph-Preserving Rolling-Window Pass (Shared Foundation)"]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "`compute_window_metrics` (above) only returns scalar summary rows — the\n",
            "`nx.Graph` built for each window is discarded once `_network_summary`/\n",
            "`_node_summary` have been extracted from it. Chapters B, C, and D below all\n",
            "need the actual graphs (or their adjacency matrices), and specifically need\n",
            "them with a **consistent node set in a consistent order** across every\n",
            "window, since `graspologic`'s multi-graph methods (`latent_position_test`,\n",
            "`OmnibusEmbed`) assume identically-shaped, identically-ordered inputs.\n",
            "\n",
            "This section reruns the windowing + dcor + graph-construction pipeline (reusing\n",
            "`generate_windows`, `network.pivot_to_wide`, `measures.compute_measure`,\n",
            "`network.build_corr_nx`, `_drop_nan_edges`, `_add_strength_attr` verbatim —\n",
            "none of these are reimplemented) but this time **keeps** each window's graph\n",
            "instead of collapsing it to a summary row. Because this duplicates the\n",
            "expensive dcor computation from scratch, it uses a coarser `step` than the\n",
            "original `CFG` (see the new `CFG_GRAPHS` below) to keep the second pass\n",
            "affordable."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def collect_window_graphs(\n",
            "    df_returns: pl.DataFrame,\n",
            "    dates: list[date],\n",
            "    cfg: EvolutionConfig,\n",
            "    *,\n",
            "    date_column: str = DATE_COL,\n",
            "    name_column: str = NAME_COL,\n",
            "    value_column: str = VALUE_COL,\n",
            ") -> tuple[dict[date, nx.Graph], dict[date, date]]:\n",
            "    \"\"\"Rerun the rolling-window dcor pipeline, keeping the pruned graph itself.\n",
            "\n",
            "    Duplicates compute_window_metrics' windowing/measure/graph-construction\n",
            "    calls (generate_windows, network.pivot_to_wide, measures.compute_measure,\n",
            "    network.build_corr_nx, _drop_nan_edges, _add_strength_attr) but returns\n",
            "    the nx.Graph per window instead of discarding it after computing scalar\n",
            "    summaries -- needed for graspologic's per-window-graph methods (Chapters\n",
            "    B/C/D below), which compute_window_metrics does not preserve.\n",
            "\n",
            "    Args:\n",
            "        df_returns: Long-format daily-returns dataframe.\n",
            "        dates: Sorted unique trading dates present in df_returns.\n",
            "        cfg: Windowing/measure/threshold parameters. Recommend a coarser\n",
            "            `step` than the main CFG (e.g. CFG_GRAPHS), since this repeats\n",
            "            the dcor computation from scratch.\n",
            "        date_column, name_column, value_column: Column names in df_returns.\n",
            "\n",
            "    Returns:\n",
            "        (graphs, window_starts): graphs maps window_end -> pruned,\n",
            "        strength-annotated nx.Graph, insertion-ordered chronologically\n",
            "        (window_end dates are strictly increasing, so no key collisions);\n",
            "        window_starts maps the same window_end keys -> window_start, kept\n",
            "        separately so _network_summary (which takes both dates) can be\n",
            "        reused unmodified in Chapter A below.\n",
            "    \"\"\"\n",
            "    windows = list(\n",
            "        generate_windows(dates, cfg.window_size, cfg.step, expanding=cfg.expanding)\n",
            "    )\n",
            "    graphs: dict[date, nx.Graph] = {}\n",
            "    window_starts: dict[date, date] = {}\n",
            "    pbar = tqdm(windows, desc=\"windows (graph-preserving)\")\n",
            "    for window_start, window_end, window_dates in pbar:\n",
            "        window_df = df_returns.filter(\n",
            "            pl.col(date_column).is_between(window_start, window_end)\n",
            "        )\n",
            "        wide = network.pivot_to_wide(window_df, date_column, name_column, value_column)\n",
            "        nodes = [c for c in wide.columns if c != date_column]\n",
            "        nodes = [n for n in nodes if wide.get_column(n).drop_nulls().len() >= 3]\n",
            "        if len(nodes) < cfg.min_nodes:\n",
            "            continue\n",
            "        measure_df = measures.compute_measure(\n",
            "            \"distance_correlation\", wide.select(nodes), nodes\n",
            "        )\n",
            "        graph = network.build_corr_nx(\n",
            "            measure_df, independent_threshold=cfg.independent_threshold\n",
            "        )\n",
            "        graph = _drop_nan_edges(graph)\n",
            "        _add_strength_attr(graph)\n",
            "        graphs[window_end] = graph\n",
            "        window_starts[window_end] = window_start\n",
            "        pbar.set_postfix(window_end=str(window_end), n_nodes=graph.number_of_nodes())\n",
            "    return graphs, window_starts"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Chapter 0 — Config: Coarser Cadence for the Graph-Preserving Pass"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "CFG_GRAPHS = EvolutionConfig(step=63)  # quarterly cadence (window unchanged at 252)\n",
            "\n",
            "n_windows_graphs = sum(\n",
            "    1\n",
            "    for _ in generate_windows(\n",
            "        dates, CFG_GRAPHS.window_size, CFG_GRAPHS.step, expanding=CFG_GRAPHS.expanding\n",
            "    )\n",
            ")\n",
            "total_dcor_calls_graphs = n_windows_graphs * n_pairs_per_window\n",
            "print(\n",
            "    f\"{n_windows_graphs} windows × {n_pairs_per_window} pairs/window = \"\n",
            "    f\"{total_dcor_calls_graphs:,} dcor calls\"\n",
            ")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Chapter 0 — Performance Estimate\n",
            "\n",
            "Reuses `n_pairs_per_window` from the original \"Pick Sane Defaults\" cell above\n",
            "(unchanged, since `CFG_GRAPHS` only changes `step`). At `step=63` (3× coarser\n",
            "than the original `step=21`), the window count and total dcor-call count\n",
            "printed above should come out to roughly **1/3** of the original full run's\n",
            "count, since the window count scales ~linearly with `1/step` for a fixed date\n",
            "range. Applying the same ~440 pairs/sec rough rate used in the original\n",
            "Performance Estimate cell to whatever `total_dcor_calls_graphs` printed above\n",
            "turns out to be: expect very roughly **one third of the original run's ~2–7\n",
            "minute estimate**, i.e. somewhere in the ~45 sec – 2.5 min range on this\n",
            "machine — **this is a rough order-of-magnitude, not a benchmark**; the exact\n",
            "call count is only known once the cell above has actually run.\n",
            "\n",
            "**Always smoke-test on a truncated date range first** (next cell), reusing\n",
            "the same `df_smoke`/`dates_smoke` truncation (pre-2010) from the original\n",
            "Smoke Test cell above."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%time graphs_smoke, window_starts_smoke = collect_window_graphs(df_smoke, dates_smoke, CFG_GRAPHS)\n",
            "print(f\"Smoke test: {len(graphs_smoke)} window graphs collected\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Chapter 0 — Full Run: Collect Window Graphs"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%time graphs_by_window, window_starts = collect_window_graphs(df_returns, dates, CFG_GRAPHS)\n",
            "print(f\"Full run: {len(graphs_by_window)} window graphs collected\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Chapter 0 — Build Common-Node Adjacency Tensor\n",
            "\n",
            "`graspologic`'s multi-graph methods (`latent_position_test`, `OmnibusEmbed`)\n",
            "require every graph to share an identical node set in an identical order.\n",
            "The DAX30 panel is *expected* to be fully balanced (all 30 constituents\n",
            "present in every window), but that must be **computed, not assumed** — the\n",
            "`<3 non-null obs` node-drop in `collect_window_graphs` could in principle\n",
            "remove different nodes in different windows. Edges are stored as **binary**\n",
            "(`weight=None`) rather than continuous dissimilarity, matching the book's\n",
            "RDPG/SBM binary-adjacency assumption used by `latent_position_test`/`ASE`/\n",
            "`OmnibusEmbed`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def build_adjacency_tensor(\n",
            "    graphs: dict[date, nx.Graph],\n",
            ") -> tuple[list[date], list[str], np.ndarray]:\n",
            "    \"\"\"Stack per-window graphs into a binary (T, n, n) adjacency tensor.\n",
            "\n",
            "    Restricts to the intersection of node sets across all collected windows\n",
            "    (computed, not assumed) and uses binary edges (weight=None) rather than\n",
            "    the continuous dissimilarity 'weight' attribute.\n",
            "\n",
            "    Args:\n",
            "        graphs: window_end -> nx.Graph, as returned by collect_window_graphs.\n",
            "\n",
            "    Returns:\n",
            "        (window_ends, common_nodes, tensor): window_ends is the sorted list\n",
            "        of window-end dates (chronological, tensor axis-0 order);\n",
            "        common_nodes is the sorted node names shared by every graph (tensor\n",
            "        axis-1/2 order); tensor has shape (T, n, n), dtype float64, values\n",
            "        in {0.0, 1.0}.\n",
            "\n",
            "    Raises:\n",
            "        ValueError: If fewer than CFG.min_nodes nodes are common to every\n",
            "            collected window.\n",
            "    \"\"\"\n",
            "    window_ends = sorted(graphs.keys())\n",
            "    node_sets = [set(graphs[we].nodes()) for we in window_ends]\n",
            "    common_nodes = sorted(set.intersection(*node_sets))\n",
            "    print(\n",
            "        f\"{len(window_ends)} windows; common node set: {len(common_nodes)} of \"\n",
            "        f\"{max(len(s) for s in node_sets)} max nodes seen in any single window\"\n",
            "    )\n",
            "    if len(common_nodes) < CFG.min_nodes:\n",
            "        raise ValueError(\n",
            "            f\"Only {len(common_nodes)} common nodes across all windows -- \"\n",
            "            f\"too few for the graspologic chapters below.\"\n",
            "        )\n",
            "    tensor = np.stack(\n",
            "        [nx.to_numpy_array(graphs[we], nodelist=common_nodes, weight=None) for we in window_ends]\n",
            "    )\n",
            "    return window_ends, common_nodes, tensor"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "window_ends, common_nodes, adj_tensor = build_adjacency_tensor(graphs_by_window)\n",
            "DHAT = int(np.ceil(np.log2(len(common_nodes))))  # shared latent dim, reused by Ch. B & D\n",
            "print(f\"Adjacency tensor shape: {adj_tensor.shape}\")\n",
            "print(f\"Shared latent dimension DHAT = {DHAT} (ceil(log2({len(common_nodes)})))"
        ]
    },
]

# Continue with all remaining cells...
# Due to size, I'll structure them in groups

# CHAPTER A
new_cells.extend([
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Chapter A: Extended Rolling Descriptive Statistics"]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "Two structural metrics not covered above:\n",
            "\n",
            "- **`transitivity`** (`nx.transitivity`) is the **global** clustering\n",
            "coefficient — the ratio of closed triplets to all triplets in the whole\n",
            "graph. This is distinct from the existing `avg_clustering`\n",
            "(`nx.average_clustering`), which is the *mean of per-node local* clustering\n",
            "coefficients; the two can diverge substantially on graphs with\n",
            "heterogeneous degree distributions.\n",
            "- **Giant-component average shortest path length** —\n",
            "`nx.average_shortest_path_length` raises on disconnected graphs, so it's\n",
            "restricted to the largest connected component. Windows whose giant\n",
            "component has fewer than 2 nodes get NaN.\n",
            "\n",
            "Both are computed **unweighted** (no `weight=` kwarg), consistent with the\n",
            "existing `avg_clustering` call above (also unweighted) and with the book's\n",
            "own usage (`Section32`/`Section33` call `nx.transitivity(G)` and\n",
            "`nx.average_shortest_path_length(G)` with no weight argument).\n",
            "\n",
            "Reuses the Chapter 0 `graphs_by_window`/`window_starts` (quarterly cadence)\n",
            "rather than re-running the windowing loop a third time."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def _extended_network_summary(graph: nx.Graph, window_end: date) -> dict:\n",
            "    \"\"\"Extra global structural metrics not covered by _network_summary.\n",
            "\n",
            "    transitivity (nx.transitivity) is the GLOBAL clustering coefficient\n",
            "    (ratio of closed triplets to all triplets), distinct from the existing\n",
            "    avg_clustering (nx.average_clustering), which is the mean of per-node\n",
            "    LOCAL clustering coefficients. giant_component_avg_shortest_path is\n",
            "    restricted to the giant (largest) connected component, since\n",
            "    nx.average_shortest_path_length raises on disconnected graphs; windows\n",
            "    whose giant component has <2 nodes get NaN. Both computed unweighted,\n",
            "    matching the existing avg_clustering's unweighted convention.\n",
            "\n",
            "    Args:\n",
            "        graph: Window's pruned, strength-annotated similarity graph.\n",
            "        window_end: Window-end date (join key back to _network_summary rows).\n",
            "\n",
            "    Returns:\n",
            "        One row: window_end, transitivity, giant_component_avg_shortest_path.\n",
            "    \"\"\"\n",
            "    transitivity = nx.transitivity(graph) if graph.number_of_nodes() else float(\"nan\")\n",
            "    components = list(nx.connected_components(graph))\n",
            "    giant = max(components, key=len) if components else set()\n",
            "    if len(giant) >= 2:\n",
            "        avg_shortest_path = nx.average_shortest_path_length(graph.subgraph(giant))\n",
            "    else:\n",
            "        avg_shortest_path = float(\"nan\")\n",
            "    return {\n",
            "        \"window_end\": window_end,\n",
            "        \"transitivity\": transitivity,\n",
            "        \"giant_component_avg_shortest_path\": avg_shortest_path,\n",
            "    }"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Chapter A — Compute Extended Metrics"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "extended_rows = []\n",
            "for window_end, graph in graphs_by_window.items():\n",
            "    window_start = window_starts[window_end]\n",
            "    row = _network_summary(graph, window_start, window_end)\n",
            "    row.update(_extended_network_summary(graph, window_end))\n",
            "    extended_rows.append(row)\n",
            "\n",
            "extended_network_metrics = pl.DataFrame(extended_rows)\n",
            "print(f\"Extended metrics: {extended_network_metrics.height} windows (quarterly cadence, CFG_GRAPHS)\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Plot (d): Extended Network-Level Metrics (Faceted Time Series)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "extended_network_long = extended_network_metrics.unpivot(\n",
            "    index=[\"window_start\", \"window_end\"],\n",
            "    on=[\n",
            "        \"n_edges\",\n",
            "        \"density\",\n",
            "        \"avg_degree\",\n",
            "        \"n_components\",\n",
            "        \"largest_component_size\",\n",
            "        \"avg_clustering\",\n",
            "        \"transitivity\",\n",
            "        \"giant_component_avg_shortest_path\",\n",
            "    ],\n",
            "    variable_name=\"metric\",\n",
            "    value_name=\"value\",\n",
            ")\n",
            "\n",
            "(\n",
            "    ggplot(extended_network_long.to_pandas(), aes(x=\"window_end\", y=\"value\"))\n",
            "    + geom_line(color=\"#2a78d6\", size=0.7)\n",
            "    + geom_point(color=\"#2a78d6\", size=0.9, alpha=0.6)\n",
            "    + facet_wrap(\"~metric\", scales=\"free_y\", ncol=2)\n",
            "    + labs(\n",
            "        x=\"Window end date\",\n",
            "        y=None,\n",
            "        title=\"DAX30 distance-correlation network: extended rolling metrics (quarterly cadence)\",\n",
            "    )\n",
            "    + theme_minimal()\n",
            "    + theme(figure_size=(10, 11))\n",
            ")"
        ]
    },
])

print(f"Added Chapter A: {len(new_cells)} total cells")
print("Continuing with Chapters B, C, D...")

# Write progress checkpoint
with open('src/tgraphportfolio/dax_network_evolution_BUILD.py', 'w') as f:
    f.write(f"# Build checkpoint: {len(new_cells)} cells built so far\n")

EOFPYTHON
