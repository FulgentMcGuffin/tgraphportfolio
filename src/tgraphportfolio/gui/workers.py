"""Background workers so the UI stays responsive during long dcor runs."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import date

from matplotlib.figure import Figure
from PySide6.QtCore import QObject, Signal, Slot
import polars as pl

from tgraphportfolio.analysis.config import PipelineConfig
from tgraphportfolio.analysis.evolution import (
    EvolutionConfig,
    compute_community_metrics,
    compute_evolution_metrics,
)
from tgraphportfolio.analysis.evolution_viz import (
    render_weighted_degree_heatmap,
    render_centrality_trajectories,
    render_extended_metrics,
    render_community_heatmap,
)
from tgraphportfolio.analysis.gui_cache import GuiDataCache
from tgraphportfolio.analysis.pipeline import PipelineResult, run_pipeline


class WorkerCancelled(Exception):
    """Raised from inside a progress/status callback to unwind a cancelled worker.

    ``run_pipeline``/``compute_evolution_metrics`` invoke their ``progress``
    callback on every pair/window with no cancellation hook of their own, so
    this is the cheapest place to interrupt a long-running computation:
    raising here propagates as a normal exception up through the (otherwise
    unmodified) computation and back into ``run()``, where it is caught
    separately from real failures.
    """


@dataclass
class EvolutionResult:
    """Artifacts produced by evolution analysis."""

    node_metrics: pl.DataFrame
    heatmap_fig: Figure
    centrality_fig: Figure
    extended_metrics_fig: Figure
    community_fig: Figure


class PipelineWorker(QObject):
    """Runs ``run_pipeline`` off the UI thread."""

    progress = Signal(int, int, str)  # done, total, description
    status = Signal(str)
    finished = Signal(object)  # PipelineResult
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        config: PipelineConfig,
        data_cache: GuiDataCache | None = None,
        edge_settings: dict | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._data_cache = data_cache
        self._edge_settings = edge_settings or {}
        self._cancel_event = cancel_event or threading.Event()
        # Will be populated during run
        self.df_returns: pl.DataFrame | None = None
        self.dates: list[date] | None = None
        # Store original date column name for reference
        self.date_column_original: str = config.date_column

    def _progress(self, done: int, total: int, desc: str) -> None:
        if self._cancel_event.is_set():
            raise WorkerCancelled()
        self.progress.emit(done, total, desc)

    def _status(self, msg: str) -> None:
        if self._cancel_event.is_set():
            raise WorkerCancelled()
        self.status.emit(msg)

    @Slot()
    def run(self) -> None:
        try:
            result: PipelineResult = run_pipeline(
                self._config,
                progress=self._progress,
                status=self._status,
                data_cache=self._data_cache,
                edge_settings=self._edge_settings,
            )
            if self._cancel_event.is_set():
                raise WorkerCancelled()

            # Store prepared data for evolution analysis
            # Get from cache if available
            from tgraphportfolio.analysis.data_access import load_table
            from tgraphportfolio.analysis.transforms import apply_transforms

            try:
                needed_cols = [
                    self._config.date_column,
                    self._config.name_column,
                    self._config.value_column,
                ]
                if self._config.filter_column:
                    needed_cols.append(self._config.filter_column)

                df = load_table(
                    self._config.db_path, self._config.table, columns=needed_cols
                )
                df = df.with_columns(
                    pl.col(self._config.date_column).cast(pl.Date, strict=False)
                )
                df = df.sort(self._config.date_column, self._config.name_column)

                if self._config.filter_column and self._config.filter_value is not None:
                    df = df.filter(
                        pl.col(self._config.filter_column).cast(pl.Utf8)
                        == self._config.filter_value
                    )

                if self._config.date_start is not None:
                    df = df.filter(
                        pl.col(self._config.date_column) >= self._config.date_start
                    )
                if self._config.date_end is not None:
                    df = df.filter(
                        pl.col(self._config.date_column) <= self._config.date_end
                    )

                df = df.drop_nulls(
                    subset=[
                        self._config.date_column,
                        self._config.name_column,
                        self._config.value_column,
                    ]
                )

                # Apply transforms
                df = apply_transforms(
                    df,
                    self._config.transforms,
                    self._config.date_column,
                    self._config.name_column,
                    [self._config.value_column],
                )

                # Normalize column names for evolution analysis
                # Rename to standard internal names to avoid column name issues
                df = df.rename(
                    {
                        self._config.date_column: "Date",
                        self._config.name_column: "Name",
                        self._config.value_column: "Close",
                    }
                )

                self.df_returns = df
                self.dates = [d for d in df.get_column("Date").unique().sort()]
            except Exception:
                pass  # Evolution worker will handle missing data gracefully

            self.finished.emit(result)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 — surface to UI
            self.failed.emit(str(exc))


class EvolutionWorker(QObject):
    """Background worker for temporal network evolution analysis."""

    progress = Signal(int, int, str)  # done, total, description
    status = Signal(str)
    finished = Signal(object)  # EvolutionResult
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        df_returns: pl.DataFrame,
        dates: list[date],
        evolution_config: EvolutionConfig,
        data_cache: GuiDataCache | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self.df_returns = df_returns
        self.dates = dates
        self.evolution_config = evolution_config
        self.data_cache = data_cache
        self._cancel_event = cancel_event or threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            self._status_wrapper("Computing evolution metrics...")

            # Compute metrics (node-level, network-level, and per-window graphs)
            # Use normalized column names (Date, Name, Close)
            metrics_result = compute_evolution_metrics(
                self.df_returns,
                self.dates,
                self.evolution_config,
                date_column="Date",
                name_column="Name",
                value_column="Close",
                progress=self._progress_wrapper,
                status=self._status_wrapper,
            )
            node_metrics = metrics_result.node_metrics
            network_metrics = metrics_result.network_metrics
            graphs = metrics_result.graphs

            if node_metrics.is_empty():
                self.failed.emit("No windows produced with minimum node count")
                return

            self._status_wrapper("Rendering weighted-degree heatmap...")
            heatmap_fig = render_weighted_degree_heatmap(node_metrics)

            self._status_wrapper("Rendering centrality trajectories...")
            n_unique_nodes = len(node_metrics["node"].unique())
            centrality_fig = render_centrality_trajectories(
                node_metrics,
                centrality_metric=self.evolution_config.centrality,
                n_nodes=min(self.evolution_config.n_top_nodes, n_unique_nodes),
            )

            self._status_wrapper("Rendering extended rolling metrics...")
            extended_metrics_fig = render_extended_metrics(network_metrics)

            self._status_wrapper("Detecting communities per window...")
            try:
                community_metrics = compute_community_metrics(
                    graphs,
                    max_clusters=self.evolution_config.max_communities,
                    min_nodes=self.evolution_config.min_nodes,
                    community_method=self.evolution_config.community_method,
                    progress=self._progress_wrapper,
                    status=self._status_wrapper,
                )
                community_fig = render_community_heatmap(community_metrics)
            except ValueError as exc:
                self._status_wrapper(f"Community detection skipped: {exc}")
                community_fig = render_community_heatmap(pl.DataFrame())

            self._status_wrapper("Evolution analysis complete.")
            self.progress.emit(1, 1, "done")

            result = EvolutionResult(
                node_metrics=node_metrics,
                heatmap_fig=heatmap_fig,
                centrality_fig=centrality_fig,
                extended_metrics_fig=extended_metrics_fig,
                community_fig=community_fig,
            )
            self.finished.emit(result)

        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Evolution analysis failed: {str(exc)}")

    def _progress_wrapper(self, done: int, total: int, desc: str) -> None:
        """Wrap progress callback; raises if cancellation was requested."""
        if self._cancel_event.is_set():
            raise WorkerCancelled()
        self.progress.emit(done, total, desc)

    def _status_wrapper(self, msg: str) -> None:
        """Wrap status callback; raises if cancellation was requested."""
        if self._cancel_event.is_set():
            raise WorkerCancelled()
        self.status.emit(msg)
