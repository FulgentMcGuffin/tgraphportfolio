"""Main window: configuration sidebar + pyvis network canvas + process log."""

from __future__ import annotations

import tempfile
from datetime import date, datetime
from pathlib import Path

from PySide6.QtCore import QDate, QThread, QUrl, Qt
from PySide6.QtGui import QColor, QPalette, QTextCursor
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from tgraphportfolio.analysis.config import PipelineConfig
from tgraphportfolio.analysis.data_access import (
    column_date_bounds,
    distinct_values,
    list_columns,
    list_tables,
)
from tgraphportfolio.analysis.measures import available_measures
from tgraphportfolio.analysis.transforms import available_transforms
from tgraphportfolio.gui.styles import APP_STYLE, BG_SIDEBAR
from tgraphportfolio.gui.workers import PipelineWorker


def _force_dark_surface(widget: QWidget, color: str = BG_SIDEBAR) -> None:
    """Ensure opaque dark fills even when platform styles ignore QSS backgrounds."""
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    widget.setAutoFillBackground(True)
    palette = widget.palette()
    qcolor = QColor(color)
    palette.setColor(QPalette.ColorRole.Window, qcolor)
    palette.setColor(QPalette.ColorRole.Base, qcolor)
    palette.setColor(QPalette.ColorRole.Button, qcolor)
    widget.setPalette(palette)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TGraph Portfolio")
        self.resize(1440, 900)
        self.setStyleSheet(APP_STYLE)

        self._db_path: Path | None = None
        self._worker_thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._temp_html: Path | None = None
        self._last_progress_line: int | None = None
        self._busy = False

        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        sidebar = self._build_sidebar()
        splitter.addWidget(sidebar)

        canvas = self._build_canvas()
        splitter.addWidget(canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 1140])

        self._set_controls_enabled(False)
        self.btn_browse.setEnabled(True)
        self._on_filter_toggled(False)

    # ------------------------------------------------------------------ UI
    def _build_sidebar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("Sidebar")
        frame.setMinimumWidth(280)
        frame.setMaximumWidth(340)
        _force_dark_surface(frame)

        outer = QVBoxLayout(frame)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        _force_dark_surface(scroll)
        if scroll.viewport() is not None:
            _force_dark_surface(scroll.viewport())
        outer.addWidget(scroll)

        content = QWidget()
        content.setObjectName("SidebarContent")
        _force_dark_surface(content)
        scroll.setWidget(content)
        form = QVBoxLayout(content)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(3)

        brand = QLabel("TGraph Portfolio")
        brand.setObjectName("Brand")
        form.addWidget(brand)

        form.addWidget(self._section("DATA SOURCE"))
        self.lbl_db = QLabel("No database selected")
        self.lbl_db.setObjectName("DbPath")
        self.lbl_db.setWordWrap(True)
        form.addWidget(self.lbl_db)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.setObjectName("SecondaryButton")
        self.btn_browse.clicked.connect(self._browse_db)
        form.addWidget(self.btn_browse)

        form.addWidget(self._section("TABLE"))
        self.cmb_table = QComboBox()
        self.cmb_table.currentTextChanged.connect(self._on_table_changed)
        form.addWidget(self.cmb_table)

        form.addWidget(self._section("DATE / DATETIME COLUMN"))
        self.cmb_date = QComboBox()
        self.cmb_date.currentTextChanged.connect(self._guess_date_range)
        form.addWidget(self.cmb_date)

        form.addWidget(self._section("NODE NAME COLUMN"))
        self.cmb_name = QComboBox()
        form.addWidget(self.cmb_name)

        form.addWidget(self._section("SERIES VALUE COLUMN"))
        self.cmb_value = QComboBox()
        form.addWidget(self.cmb_value)

        form.addWidget(self._section("OPTIONAL FILTER"))
        self.chk_filter = QCheckBox("Filter by column value")
        self.chk_filter.toggled.connect(self._on_filter_toggled)
        form.addWidget(self.chk_filter)
        self.cmb_filter_col = QComboBox()
        self.cmb_filter_col.currentTextChanged.connect(self._load_filter_values)
        form.addWidget(self.cmb_filter_col)
        self.cmb_filter_val = QComboBox()
        self.cmb_filter_val.setEditable(True)
        form.addWidget(self.cmb_filter_val)

        form.addWidget(self._section("TRANSFORMS (ORDERED)"))
        self.lst_transforms = QListWidget()
        for transform_id, label in available_transforms():
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, transform_id)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(Qt.CheckState.Checked)
            self.lst_transforms.addItem(item)
        self.lst_transforms.setMaximumHeight(56)
        form.addWidget(self.lst_transforms)
        tf_row = QHBoxLayout()
        tf_row.setSpacing(4)
        self.btn_tf_up = QPushButton("Up")
        self.btn_tf_up.setObjectName("SecondaryButton")
        self.btn_tf_up.clicked.connect(lambda: self._move_transform(-1))
        self.btn_tf_down = QPushButton("Down")
        self.btn_tf_down.setObjectName("SecondaryButton")
        self.btn_tf_down.clicked.connect(lambda: self._move_transform(1))
        tf_row.addWidget(self.btn_tf_up)
        tf_row.addWidget(self.btn_tf_down)
        form.addLayout(tf_row)

        form.addWidget(self._section("CONNECTION MEASURE"))
        self.cmb_measure = QComboBox()
        for measure_id, label in available_measures():
            self.cmb_measure.addItem(label, measure_id)
        form.addWidget(self.cmb_measure)

        form.addWidget(self._section("DATE RANGE"))
        dates = QHBoxLayout()
        dates.setSpacing(4)
        self.date_start = QDateEdit()
        self.date_start.setCalendarPopup(True)
        self.date_start.setDisplayFormat("yyyy-MM-dd")
        self.date_end = QDateEdit()
        self.date_end.setCalendarPopup(True)
        self.date_end.setDisplayFormat("yyyy-MM-dd")
        dates.addWidget(self.date_start)
        dates.addWidget(self.date_end)
        form.addLayout(dates)

        form.addWidget(self._section("INDEPENDENCE THRESHOLD"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.0, 1.0)
        self.spin_threshold.setSingleStep(0.01)
        self.spin_threshold.setValue(0.33)
        form.addWidget(self.spin_threshold)

        form.addSpacing(8)
        self.btn_run = QPushButton("Build network")
        self.btn_run.clicked.connect(self._run_pipeline)
        form.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        form.addWidget(self.progress)

        self.lbl_status = QLabel("Select a DuckDB / SQLite file to begin.")
        self.lbl_status.setObjectName("StatusLabel")
        self.lbl_status.setWordWrap(True)
        form.addWidget(self.lbl_status)

        form.addStretch(1)
        return frame

    def _build_canvas(self) -> QWidget:
        wrap = QFrame()
        wrap.setObjectName("CanvasFrame")
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        canvas = QFrame()
        canvas.setObjectName("Canvas")
        canvas_layout = QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.web = QWebEngineView()
        self.web.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.web.setHtml(self._placeholder_html())
        canvas_layout.addWidget(self.web)
        layout.addWidget(canvas, stretch=3)

        log_title = QLabel("PROCESS LOG")
        log_title.setObjectName("LogTitle")
        layout.addWidget(log_title)

        self.process_log = QPlainTextEdit()
        self.process_log.setObjectName("ProcessLog")
        self.process_log.setReadOnly(True)
        self.process_log.setMaximumBlockCount(5000)
        self.process_log.setPlaceholderText(
            "Pipeline status, transforms, and pair-progress will appear here…"
        )
        self.process_log.setMinimumHeight(140)
        self.process_log.setMaximumHeight(220)
        layout.addWidget(self.process_log, stretch=0)
        return wrap

    @staticmethod
    def _section(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("SectionTitle")
        return label

    @staticmethod
    def _placeholder_html() -> str:
        return """
        <!DOCTYPE html>
        <html><body style="margin:0;height:100vh;display:flex;align-items:center;
        justify-content:center;background:#0f172a;color:#94a3b8;
        font-family:Segoe UI,Helvetica Neue,sans-serif;">
        <div style="text-align:center;max-width:28rem;padding:2rem;">
          <div style="font-size:1.35rem;font-weight:700;color:#ffffff;margin-bottom:.5rem;">
            Network canvas
          </div>
          <div style="line-height:1.5;">
            Configure the data source and options on the left, then click
            <b style="color:#60a5fa;">Build network</b> to render the interactive graph.
          </div>
        </div></body></html>
        """

    @staticmethod
    def _building_html() -> str:
        return """
        <!DOCTYPE html>
        <html><body style="margin:0;height:100vh;display:flex;align-items:center;
        justify-content:center;background:#0f172a;color:#94a3b8;
        font-family:Segoe UI,Helvetica Neue,sans-serif;">
        <div style="text-align:center;padding:2rem;">
          <div style="font-size:1.2rem;font-weight:600;color:#7dd3fc;margin-bottom:.4rem;">
            Building network…
          </div>
          <div style="line-height:1.5;color:#94a3b8;">
            Previous view cleared. Progress is shown in the process log.
          </div>
        </div></body></html>
        """

    # ------------------------------------------------------------- logging
    def _append_log(self, message: str, *, replace_last: bool = False) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        if replace_last and self._last_progress_line is not None:
            cursor = self.process_log.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deletePreviousChar()  # remove leftover newline
            self.process_log.setTextCursor(cursor)
        self.process_log.appendPlainText(line)
        self.process_log.moveCursor(QTextCursor.MoveOperation.End)
        if replace_last:
            self._last_progress_line = self.process_log.blockCount()
        else:
            self._last_progress_line = None

    # ----------------------------------------------------------- data wiring
    def _browse_db(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select database",
            str(Path.home()),
            "Databases (*.duckdb *.db *.sqlite *.sqlite3);;All files (*.*)",
        )
        if not path:
            return
        self._db_path = Path(path)
        self.lbl_db.setText(str(self._db_path))
        try:
            tables = list_tables(self._db_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Database error", str(exc))
            return
        self.cmb_table.blockSignals(True)
        self.cmb_table.clear()
        self.cmb_table.addItems(tables)
        self.cmb_table.blockSignals(False)
        self._set_controls_enabled(True)
        if tables:
            self._on_table_changed(tables[0])
        msg = f"Loaded database with {len(tables)} table(s)."
        self.lbl_status.setText(msg)
        self._append_log(msg)

    def _on_table_changed(self, table: str) -> None:
        if not self._db_path or not table:
            return
        try:
            columns = list_columns(self._db_path, table)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Schema error", str(exc))
            return

        for combo in (self.cmb_date, self.cmb_name, self.cmb_value, self.cmb_filter_col):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(columns)
            combo.blockSignals(False)

        self._guess_roles(columns)
        self._load_filter_values(self.cmb_filter_col.currentText())
        self._guess_date_range()
        self.btn_run.setEnabled(True)
        self._append_log(f"Selected table {table!r} ({len(columns)} columns).")

    def _guess_roles(self, columns: list[str]) -> None:
        lower = {c.lower(): c for c in columns}

        def pick(candidates: list[str], combo: QComboBox, fallback_idx: int = 0) -> None:
            for name in candidates:
                if name in lower:
                    combo.setCurrentText(lower[name])
                    return
            if combo.count():
                combo.setCurrentIndex(min(fallback_idx, combo.count() - 1))

        pick(["date", "index", "datetime", "timestamp"], self.cmb_date)
        pick(["name", "stock", "ticker", "symbol"], self.cmb_name)
        pick(["close", "adj_close", "price", "value"], self.cmb_value)
        pick(["eqindex", "index_name", "universe"], self.cmb_filter_col)

    def _on_filter_toggled(self, enabled: bool) -> None:
        self.cmb_filter_col.setEnabled(enabled)
        self.cmb_filter_val.setEnabled(enabled)
        if enabled:
            self._load_filter_values(self.cmb_filter_col.currentText())

    def _load_filter_values(self, column: str) -> None:
        if not self._db_path or not column or not self.chk_filter.isChecked():
            return
        table = self.cmb_table.currentText()
        if not table:
            return
        try:
            values = distinct_values(self._db_path, table, column)
        except Exception:  # noqa: BLE001
            values = []
        current = self.cmb_filter_val.currentText()
        self.cmb_filter_val.blockSignals(True)
        self.cmb_filter_val.clear()
        self.cmb_filter_val.addItems(values)
        if current:
            self.cmb_filter_val.setEditText(current)
        self.cmb_filter_val.blockSignals(False)

    def _guess_date_range(self, *_args) -> None:
        """Seed date widgets from the selected date column when possible."""
        today = QDate.currentDate()
        self.date_end.setDate(today)
        self.date_start.setDate(QDate(2010, 1, 1))
        if not self._db_path:
            return
        table = self.cmb_table.currentText()
        date_col = self.cmb_date.currentText()
        if not table or not date_col:
            return
        try:
            lo, hi = column_date_bounds(self._db_path, table, date_col)
        except Exception:  # noqa: BLE001
            return
        if lo is not None:
            self.date_start.setDate(QDate(lo.year, lo.month, lo.day))
        if hi is not None:
            self.date_end.setDate(QDate(hi.year, hi.month, hi.day))

    def _move_transform(self, delta: int) -> None:
        row = self.lst_transforms.currentRow()
        if row < 0:
            return
        new_row = row + delta
        if new_row < 0 or new_row >= self.lst_transforms.count():
            return
        item = self.lst_transforms.takeItem(row)
        self.lst_transforms.insertItem(new_row, item)
        self.lst_transforms.setCurrentRow(new_row)

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable sidebar controls. Browse stays available unless busy."""
        for widget in (
            self.btn_browse,
            self.cmb_table,
            self.cmb_date,
            self.cmb_name,
            self.cmb_value,
            self.chk_filter,
            self.lst_transforms,
            self.btn_tf_up,
            self.btn_tf_down,
            self.cmb_measure,
            self.date_start,
            self.date_end,
            self.spin_threshold,
            self.btn_run,
        ):
            widget.setEnabled(enabled)
        self.cmb_filter_col.setEnabled(enabled and self.chk_filter.isChecked())
        self.cmb_filter_val.setEnabled(enabled and self.chk_filter.isChecked())
        # Run only makes sense once a DB is loaded.
        if enabled and self._db_path is None:
            self.btn_run.setEnabled(False)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._set_controls_enabled(not busy)

    # -------------------------------------------------------------- run
    def _selected_transforms(self) -> list[str]:
        transforms: list[str] = []
        for i in range(self.lst_transforms.count()):
            item = self.lst_transforms.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                transforms.append(item.data(Qt.ItemDataRole.UserRole))
        return transforms

    def _build_config(self) -> PipelineConfig:
        if self._db_path is None:
            raise ValueError("No database selected.")

        filter_column = None
        filter_value = None
        if self.chk_filter.isChecked():
            filter_column = self.cmb_filter_col.currentText() or None
            filter_value = self.cmb_filter_val.currentText() or None

        start = self.date_start.date().toPython()
        end = self.date_end.date().toPython()
        if isinstance(start, date) and isinstance(end, date) and start > end:
            raise ValueError("Start date must be on or before end date.")

        value_col = self.cmb_value.currentText()
        measure_id = self.cmb_measure.currentData()
        title = f"{self.cmb_measure.currentText()} ({value_col})"

        return PipelineConfig(
            db_path=self._db_path,
            table=self.cmb_table.currentText(),
            date_column=self.cmb_date.currentText(),
            name_column=self.cmb_name.currentText(),
            value_column=value_col,
            filter_column=filter_column,
            filter_value=filter_value,
            transforms=self._selected_transforms(),
            measure=measure_id,
            date_start=start,
            date_end=end,
            independent_threshold=float(self.spin_threshold.value()),
            title=title,
        )

    def _run_pipeline(self) -> None:
        if self._busy:
            return
        try:
            config = self._build_config()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return

        self._set_busy(True)
        self.progress.setValue(0)
        self._last_progress_line = None
        self.process_log.clear()
        self.lbl_status.setText("Starting…")
        self._append_log("Starting pipeline…")
        # Clear any previously rendered network before the new run.
        self.web.setHtml(self._building_html())

        # Keep strong Python refs — a local worker is GC'd and the thread dies
        # before run() executes (see "QThread: Destroyed while thread is still running").
        self._worker_thread = QThread(self)
        self._worker = PipelineWorker(config)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker_thread.start()

    def _on_status(self, message: str) -> None:
        self.lbl_status.setText(message)
        self._append_log(message)

    def _on_progress(self, done: int, total: int, desc: str) -> None:
        if total <= 0:
            return
        pct = int(100 * done / total)
        self.progress.setValue(pct)
        # Refresh the log line periodically so the UI stays responsive.
        if done not in (0, total) and done % max(1, total // 100) != 0:
            self.lbl_status.setText(f"Computing pairs: {desc}")
            return
        bar_w = 24
        filled = int(bar_w * done / total)
        bar = "█" * filled + "░" * (bar_w - filled)
        self._append_log(
            f"dcor {bar} {pct:3d}% ({done}/{total})  {desc}",
            replace_last=True,
        )
        self.lbl_status.setText(f"Computing pairs: {desc}")

    def _cleanup_worker(self) -> None:
        self._worker = None
        self._worker_thread = None

    def _on_finished(self, html: str) -> None:
        self.progress.setValue(100)
        self.lbl_status.setText("Network ready.")
        self._append_log("Network ready.")
        tmp = tempfile.NamedTemporaryFile(
            prefix="tgraph_", suffix=".html", delete=False
        )
        tmp.write(html.encode("utf-8"))
        tmp.close()
        if self._temp_html and self._temp_html.exists():
            try:
                self._temp_html.unlink()
            except OSError:
                pass
        self._temp_html = Path(tmp.name)
        self.web.load(QUrl.fromLocalFile(str(self._temp_html.resolve())))
        self._cleanup_worker()
        self._set_busy(False)

    def _on_failed(self, message: str) -> None:
        self.progress.setValue(0)
        self.lbl_status.setText("Failed.")
        self._append_log(f"ERROR: {message}")
        self._cleanup_worker()
        self._set_busy(False)
        QMessageBox.critical(self, "Pipeline failed", message)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(2000)
        if self._temp_html and self._temp_html.exists():
            try:
                self._temp_html.unlink()
            except OSError:
                pass
        super().closeEvent(event)
