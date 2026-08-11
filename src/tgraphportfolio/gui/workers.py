"""Background workers so the UI stays responsive during long dcor runs."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from tgraphportfolio.analysis.config import PipelineConfig
from tgraphportfolio.analysis.gui_cache import GuiDataCache
from tgraphportfolio.analysis.pipeline import PipelineResult, run_pipeline


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
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 — surface to UI
            self.failed.emit(str(exc))
