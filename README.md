## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```bash
uv sync
uv run tgraph-gui
```

In the GUI, point the sidebar at a DuckDB database, choose columns / filters, then click **Build Network**.

| Distance correlation network (DAX30 close) | Degree distribution (CAC40 close) |
|:---:|:---:|
| ![Distance correlation network for DAX30 closing price returns](rsrc/images/dax_dcor.png) | ![Degree distribution for DAX30 closing price returns](rsrc/images/cac_degrees.png) |

