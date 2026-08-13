# TGraph Portfolio

## Overview
The goal of this project is to provide deep insights into the temporal evolution of network structures derived from relationship patterns in multi-asset panel data over time. By transforming static cross-sectional relationships into a chronological sequence of networks (a time series of graphs), the platform exposes the underlying structural shifts, transition dynamics, and regime changes in multivariate time series.

While the default analysis utilizes equity market panel datasets (such as DAX30 or CAC40 price returns), the system is built on a generalized, abstract database backend. It can be applied directly to **any multivariate time-series panel dataset** that has a temporal index, a node identifier (representing entities such as stocks, sensors, assets, or regions), and numeric series values.

### Methods Covered
The toolkit integrates several statistical and network analysis techniques:

1. **Pluggable Data Transforms**: Performs customizable node-wise preprocessing operations, such as computing daily simple returns, to prepare the panel data for connection modeling.
2. **Selectable Pairwise Connection Measures**:
   - **Distance Correlation**: Captures both linear and non-linear associations without assuming monotonic relationships.
   - **Pearson Correlation**: Evaluates standard linear correlation.
   - **Spearman Correlation**: Measures monotonic rank-based relationships.
3. **Graph Construction and Thresholding**: Prunes weak connections using a user-defined independence threshold to build unweighted or weighted `NetworkX` graphs.
4. **Dynamic Network Visualization**: Renders interactive, physics-simulated network graphs utilizing `pyvis` to explore node connections dynamically.
5. **Static Descriptive Metrics**: Computes and labels extreme nodes (minimum, maximum, and mode degrees) on degree-distribution histograms styled for dark-mode visualization.
6. **Temporal Evolution Tracking**:
   - **Weighted Degree Heatmaps**: Visualizes how individual node connectivity changes chronologically across rolling or expanding windows.
   - **Centrality Trajectories**: Plots eigenvector, betweenness, or degree centrality of the most variable nodes over time to identify systemic shifts.
   - **Regime and Change-Point Detection**: Employs `graspologic`'s `latent_position_test` to statistically test whether consecutive window network snapshots share the same latent positions, flagging significant structural transitions with Holm-Bonferroni correction.
   - **Latent Trajectories**: Models dynamic community tracking and multi-graph latent position embedding via Joint/Omnibus Spectral Embedding.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```bash
uv sync
uv run tgraph-gui
```

In the GUI, point the sidebar at a DuckDB or SQLite database, choose columns / filters, then click **Build Network**.

| Distance correlation network (DAX30 close) | Degree distribution (CAC40 close) |
|:---:|:---:|
| ![Distance correlation network for DAX closing price returns](rsrc/images/dax_dcor.png) | ![Degree distribution for CAC closing price returns](rsrc/images/cac_degrees.png) |

| Extended rolling metrics (CAC40 close) | Weighted-degree evolution (HSI50 close) |
|:---:|:---:|
| ![Extended rolling network metrics for CAC40 closing price returns](rsrc/images/cac_extended_rolling.png) | ![Node weighted-degree heatmap for HSI50 closing price returns](rsrc/images/hsi_weighted_degrees.png) |

| Centrality trajectories (HSI50 close) | Community membership evolution (CAC40 close) |
|:---:|:---:|
| ![Eigenvector centrality trajectories for HSI50 closing price returns](rsrc/images/hsi_centrality.png) | ![Node community membership heatmap for CAC40 closing price returns](rsrc/images/cac_community.png) |

## References

### Project Foundations & Libraries
* **Hedgecraft**: The portfolio management algorithm and network analysis foundations: [GitHub Repository](https://github.com/mayabenowitz/Hedgecraft).
* **graspologic (Python 3.13 Compatible Fork)**: Graph statistical algorithms optimized for modern Python and dependency stacks: [GitHub Repository](https://github.com/FulgentMcGuffin/graspologic).
* **NetworkX**: Network analysis and graph structures: [Official Site](https://networkx.org/) | [GitHub](https://github.com/networkx/networkx).
* **pyvis**: Interactive HTML-based network visualizations: [Documentation](https://pyvis.readthedocs.io/) | [GitHub](https://github.com/WestHealth/pyvis).
* **Polars**: High-performance, multi-threaded dataframe execution engine: [Documentation](https://docs.pola.rs/) | [GitHub](https://github.com/pola-rs/polars).

### Statistical Relationships & Correlation
* **Distance Correlation**: Capturing linear and non-linear association: [Wikipedia](https://en.wikipedia.org/wiki/Distance_correlation).
* **Pearson Correlation Coefficient**: Evaluating linear correlation: [Wikipedia](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
* **Spearman's Rank Correlation Coefficient**: Monotonic relationship strength: [Wikipedia](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient).

### Multiple Hypothesis Testing
* **Holm-Bonferroni Method**: Sequentially rejective procedure controlling family-wise error rates: [Wikipedia](https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method).
* **Foundational Paper**: Holm, S. (1979). *"A simple sequentially rejective multiple test procedure."* Scandinavian Journal of Statistics, 6(2), 65-70: [DOI (Wiley/JSTOR)](https://doi.org/10.2307/4615733).

### Multi-Graph Latent Position Embedding
* **Omnibus Embedding Tutorial**: Simultaneously embedding multiple matched-vertex graphs into a common canonical coordinate system: [graspologic Tutorial](https://graspologic-org.github.io/graspologic/tutorials/embedding/Omnibus.html).
* **Foundational Paper**: Levin, K., Athreya, A., Tang, M., Lyzinski, V., & Priebe, C. E. (2017). *"A central limit theorem for an omnibus embedding of multiple random graphs and implications for multiscale network inference."* arXiv preprint arXiv:1705.09355: [arXiv Paper](https://arxiv.org/abs/1705.09355).

