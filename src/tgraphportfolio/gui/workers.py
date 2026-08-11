"""Background workers so the UI stays responsive during long dcor runs."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from tgraphportfolio.analysis.config import PipelineConfig
from tgraphportfolio.analysis.pipeline import run_pipeline


class PipelineWorker(QObject):
    """Runs ``run_pipeline`` off the UI thread."""

    progress = Signal(int, int, str)  # done, total, description
    status = Signal(str)
    finished = Signal(str)  # html
    failed = Signal(str)

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__()
        self._config = config

    @Slot()
    def run(self) -> None:
        try:
            html = run_pipeline(
                self._config,
                progress=lambda done, total, desc: self.progress.emit(
                    done, total, desc
                ),
                status=lambda msg: self.status.emit(msg),
            )
            self.finished.emit(html)
        except Exception as exc:  # noqa: BLE001 — surface to UI
            self.failed.emit(str(exc))
