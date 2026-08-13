"""Background workers so the UI stays responsive during long dcor runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from PySide6.QtCore import QObject, Signal, Slot
import polars as pl

from tgraphportfolio.analysis.config import PipelineConfig
from tgraphportfolio.analysis.evolution import EvolutionConfig, compute_evolution_metrics
from tgraphportfolio.analysis.evolution_viz import (
    render_weighted_degree_heatmap,
    render_centrality_trajectories,
)
from tgraphportfolio.analysis.gui_cache import GuiDataCache
from tgraphportfolio.analysis.pipeline import PipelineResult, run_pipeline


@dataclass
class EvolutionResult:
    """Artifacts produced by evolution analysis."""

    node_metrics: pl.DataFrame
    heatmap_png: bytes
    centrality_png: bytes


class PipelineWorker(QObject):
    """Runs ``run_pipeline`` off the UI thread."""

    progress = Signal(int, int, str)  # done, total, description
    status = Signal(str)
    finished = Signal(object)  # PipelineResult
    failed = Signal(str)

    def __init__(
        self,
        config: PipelineConfig,
        data_cache: GuiDataCache | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._data_cache = data_cache
        # Will be populated during run
        self.df_returns: pl.DataFrame | None = None
        self.dates: list[date] | None = None
        # Store original date column name for reference
        self.date_column_original: str = config.date_column

    @Slot()
    def run(self) -> None:
        try:
            result: PipelineResult = run_pipeline(
                self._config,
                progress=lambda done, total, desc: self.progress.emit(
                    done, total, desc
                ),
                status=lambda msg: self.status.emit(msg),
                data_cache=self._data_cache,
            )
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

                df = load_table(self._config.db_path, self._config.table, columns=needed_cols)
                df = df.with_columns(pl.col(self._config.date_column).cast(pl.Date, strict=False))
                df = df.sort(self._config.date_column, self._config.name_column)

                if self._config.filter_column and self._config.filter_value is not None:
                    df = df.filter(
                        pl.col(self._config.filter_column).cast(pl.Utf8) == self._config.filter_value
                    )

                if self._config.date_start is not None:
                    df = df.filter(pl.col(self._config.date_column) >= self._config.date_start)
                if self._config.date_end is not None:
                    df = df.filter(pl.col(self._config.date_column) <= self._config.date_end)

                df = df.drop_nulls(subset=[
                    self._config.date_column,
                    self._config.name_column,
                    self._config.value_column,
                ])

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
                df = df.rename({
                    self._config.date_column: "Date",
                    self._config.name_column: "Name",
                    self._config.value_column: "Close",
                })

                self.df_returns = df
                self.dates = [d for d in df.get_column("Date").unique().sort()]
            except Exception:
                pass  # Evolution worker will handle missing data gracefully

            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 — surface to UI
            self.failed.emit(str(exc))


class EvolutionWorker(QObject):
    """Background worker for temporal network evolution analysis."""

    progress = Signal(int, int, str)  # done, total, description
    status = Signal(str)
    finished = Signal(object)  # EvolutionResult
    failed = Signal(str)

    def __init__(
        self,
        df_returns: pl.DataFrame,
        dates: list[date],
        evolution_config: EvolutionConfig,
        data_cache: GuiDataCache | None = None,
    ) -> None:
        super().__init__()
        self.df_returns = df_returns
        self.dates = dates
        self.evolution_config = evolution_config
        self.data_cache = data_cache

    @Slot()
    def run(self) -> None:
        try:
            self.status.emit("Computing evolution metrics...")

            # Compute metrics
            # Use normalized column names (Date, Name, Close)
            node_metrics = compute_evolution_metrics(
                self.df_returns,
                self.dates,
                self.evolution_config,
                date_column="Date",
                name_column="Name",
                value_column="Close",
                progress=self._progress_wrapper,
                status=self.status.emit,
            )

            if node_metrics.is_empty():
                self.failed.emit("No windows produced with minimum node count")
                return

            self.status.emit("Rendering weighted-degree heatmap...")
            heatmap_png = render_weighted_degree_heatmap(node_metrics)

            self.status.emit("Rendering centrality trajectories...")
            n_unique_nodes = len(node_metrics["node"].unique())
            centrality_png = render_centrality_trajectories(
                node_metrics,
                centrality_metric=self.evolution_config.centrality,
                n_nodes=min(10, n_unique_nodes),
            )

            self.status.emit("Evolution analysis complete.")
            self.progress.emit(1, 1, "done")

            result = EvolutionResult(
                node_metrics=node_metrics,
                heatmap_png=heatmap_png,
                centrality_png=centrality_png,
            )
            self.finished.emit(result)

        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Evolution analysis failed: {str(exc)}")

    def _progress_wrapper(self, done: int, total: int, desc: str) -> None:
        """Wrap progress callback."""
        self.progress.emit(done, total, desc)
