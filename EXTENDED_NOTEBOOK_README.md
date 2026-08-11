# DAX30 Network Evolution Notebook: Extended Analysis

## Overview

The `src/tgraphportfolio/dax_network_evolution.ipynb` notebook has been extended from **32 cells** to **75 cells** with **4 new chapters** that apply techniques from *Hands-On Network Machine Learning* (Bridgeford/Loftus/Vogelstein) to temporal network analysis.

The original 32 cells (untouched) compute rolling-window distance-correlation networks for DAX30 constituent stocks, tracking scalar network summary metrics over time. The new 43 cells add structural analysis, regime detection, dynamic communities, and node trajectories—all built on `graspologic`.

## What Each Chapter Does

### **Chapter 0: Graph-Preserving Rolling-Window Pass** (Foundation)

Reruns the windowing + distance-correlation computation pipeline at a **coarser cadence** (quarterly, `step=63` vs. the original monthly `step=21`), this time keeping the actual `nx.Graph` objects instead of discarding them after extracting summary rows. This is necessary because `graspologic`'s multi-graph methods (`latent_position_test`, `OmnibusEmbed`) require:
- Identical node sets across all windows (computed via intersection)
- Identical ordering (sorted alphabetically)
- Binary adjacency matrices (`weight=None`)

**Output**: Adjacency tensor of shape `(T, n, n)` where T ≈ 44 windows, n = 30 DAX stocks.

### **Chapter A: Extended Rolling Descriptive Statistics**

Adds two more global structural metrics (beyond the original 6):
- **Transitivity**: Global clustering coefficient (ratio of closed triplets to all triplets)
- **Giant-component average shortest path**: Mean distance between nodes, restricted to the largest connected component

**New plot (d)**: Faceted line plot with 8 metrics over time.

### **Chapter B: Regime / Change-Point Detection**

Performs pairwise hypothesis tests between consecutive window snapshots using `graspologic.inference.latent_position_test`, testing the null hypothesis that latent positions (RDPG parameters) are identical. Small p-values flag structural shifts.

**Multiple comparisons correction**: Hand-rolled Holm-Bonferroni method (avoids adding `statsmodels` as a dependency).

**Windows caveat**: `latent_position_test` with `workers=-1` uses multiprocessing. Jupyter under Windows may experience spawn-vs-fork semantics issues. Smoke test on 5 pairs first; fall back to `workers=1` if it hangs.

**New plot (e)**: Bar chart of −log₁₀(Holm-adjusted p-value) per transition, with significance threshold.

### **Chapter C: Dynamic Community Detection & Drift Tracking**

Per-window community detection via:
1. **Adjacency Spectral Embedding (ASE)**: Embeds each window's adjacency matrix independently
2. **KMeans clustering**: Auto-selects k via silhouette score

Tracks community-label drift using **Adjusted Rand Index (ARI)** — a permutation-invariant label-agreement measure.

**Design note**: ARI is computed directly on per-window labels with no post-hoc remapping (ARI is label-invariant by construction). The heatmap visualization uses `remap_labels` *purely for visual continuity* (labeling aid), explicitly flagged as non-rigorous.

**New plots**:
- **(f)**: ARI over time (line chart)
- **(g)**: Node × window community-membership heatmap (categorical coloring)

### **Chapter D: Node Trajectory Embedding via Omnibus**

Unlike Chapter C (independent per-window ASE fits → unrelated coordinate systems), `graspologic.embed.OmnibusEmbed` jointly embeds **all** T windows into a single shared, pre-aligned latent space. This allows direct visualization of node trajectories without Procrustes alignment.

Selects the **top 6 most-moving nodes** (ranked by total latent-space displacement across time) for legible visualization.

**New plot (h)**: 2D trajectory plot in latent space, with paths colored by node and an arrowhead indicating direction over time.

## Running the Notebook

### Prerequisites

`graspologic` is installed by `uv sync` from the [Python 3.13 / NumPy 2–compatible fork](https://github.com/FulgentMcGuffin/graspologic). Same environment as the GUI (`uv run tgraph-gui`).

### Recommended Workflow

1. **Run Chapters 0–A first** (graph collection, extended stats)
   - These are fast (< 5 min total)
   - Verify that `build_adjacency_tensor` reports 30 common nodes and `DHAT = 5`

2. **Chapter B smoke test** (5-pair regime detection)
   - Takes ~30 sec
   - If `workers=-1` hangs, edit the full run cell to use `workers=1`
   - Full run then takes 2–3 min

3. **Chapters C & D** (communities and trajectories)
   - Fast (< 2 min each)
   - Verify that community counts are in a reasonable range (2–10)
   - Check that trajectory plot doesn't show all nodes overlapping at the origin (would indicate degenerate embedding)

### Clearing Outputs Before Committing

All cells (original 32 + new 43) produce outputs that must be cleared before git commit:

```
Jupyter → Kernel → Restart & Clear All Outputs → Save (Ctrl+S)
```

This mirrors the existing notebook's convention (repeated in the new final markdown cell).

## Implementation Notes

### Notebook Structure
- **Cells 0–31**: Original (untouched)
- **Cells 32–34**: Preamble + imports
- **Cells 35–46**: Chapter 0 (graph-preserving pass)
- **Cells 47–51**: Chapter A (extended stats)
- **Cells 52–59**: Chapter B (regime detection)
- **Cells 60–67**: Chapter C (community dynamics)
- **Cells 68–74**: Chapter D (node trajectories)
- **Cell 75**: Final reminder

### Key Design Decisions

1. **Binary adjacency matrices** (`weight=None`)
   - Matches RDPG/SBM assumption in the book
   - Discards the original continuous dissimilarity weights
   - Compatible with `latent_position_test`, `AdjacencySpectralEmbed`, `OmnibusEmbed`

2. **Shared latent dimension**
   - `DHAT = ceil(log2(n_nodes))` = 5 for n=30
   - Consistent across Chapters B (regime detection) and D (Omnibus embedding)

3. **Quarterly cadence for Chapter 0**
   - `step=63` vs. original `step=21`
   - Reduces dcor-call count to ~1/3 of the original run
   - Keeps the graph-recomputation cost affordable (expect ~1–2 min)

4. **Hand-rolled Holm-Bonferroni** (no `statsmodels`)
   - Implemented step-down algorithm directly
   - Avoids adding a second large dependency

5. **Per-window ARI** (label-invariant)
   - No Procrustes alignment between community labels
   - Label remapping in the heatmap is visual only (non-rigorous)

## Verification Checklist

After running the full notebook, verify:

- [ ] Chapter 0:
  - `build_adjacency_tensor` reports 30 common nodes
  - `adj_tensor.shape == (T, 30, 30)` where T ≈ 44
  - `DHAT = 5`

- [ ] Chapter A:
  - 8-facet plot renders without all-NaN facets
  - `giant_component_avg_shortest_path` is NaN only for highly fragmented windows

- [ ] Chapter B:
  - Smoke test (5 pairs) runs without hanging
  - Full run completes in 2–5 min
  - Regime changes: non-trivial number of transitions significant (not all 0, not all T-1)

- [ ] Chapter C:
  - `cluster_counts` prints a reasonable range (roughly 2–10)
  - ARI values are in [−1, 1], mostly positive
  - Heatmap renders (no palette overflow if max community ID < 6)

- [ ] Chapter D:
  - `latent_tensor.shape == (T, 30, DHAT)` = (44, 30, 5)
  - Trajectory plot shows visibly different path shapes for top 6 nodes (not overlapping at origin)

## Related Files

- `pyproject.toml`: `graspologic` from [FulgentMcGuffin/graspologic](https://github.com/FulgentMcGuffin/graspologic) (Python 3.13 / NumPy 2)
- `src/tgraphportfolio/dax_network_evolution.ipynb`: Main notebook (75 cells)
- This file: `EXTENDED_NOTEBOOK_README.md`

## Future Work

Possible extensions (not implemented):
- Multi-resolution community detection (varying k per window)
- Procrustes alignment of per-window ASE coordinates for Chapter C
- Temporal smoothing of ARI values (lowess, rolling mean)
- Node importance ranking based on trajectory variance
- Integration of additional graph features (e.g., motifs, graphlets) into regime detection
