"""Main window: configuration sidebar + pyvis network canvas + process log."""

from __future__ import annotations

import tempfile
from datetime import date, datetime
from pathlib import Path

from PySide6.QtCore import QDate, QThread, QUrl, Qt
from PySide6.QtGui import QColor, QPalette, QPixmap, QTextCursor
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
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
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from tgraphportfolio.analysis.config import PipelineConfig
from tgraphportfolio.analysis.data_access import (
    column_date_bounds,
    distinct_values,
    list_columns,
    list_tables,
)
from tgraphportfolio.analysis.evolution import EvolutionConfig
from tgraphportfolio.analysis.gui_cache import GuiDataCache
from tgraphportfolio.analysis.measures import (
    ACE_AVAILABLE,
    ACE_IMPORT_ERROR,
    available_measures,
    measure_short_label,
)
from tgraphportfolio.analysis.pipeline import PipelineResult
from tgraphportfolio.analysis.transforms import available_transforms
from tgraphportfolio.gui.evolution_settings_dialog import EvolutionSettingsDialog
from tgraphportfolio.gui.styles import APP_STYLE, BG_SIDEBAR
from tgraphportfolio.gui.workers import EvolutionWorker, EvolutionResult, PipelineWorker


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
        self.resize(1440, 990)  # ~10% taller so histogram title/xlabel fit
        self.setStyleSheet(APP_STYLE)

        self._db_path: Path | None = None
        self._worker_thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._temp_html: Path | None = None
        self._last_progress_line: int | None = None
        self._busy = False
        # Session-scoped Polars memoization (in-memory SQLite via framecache).
        self._data_cache = GuiDataCache.create()

        # Evolution analysis state
        self._evolution_config = EvolutionConfig()
        self._evolution_worker_thread: QThread | None = None
        self._evolution_worker: EvolutionWorker | None = None
        self._current_n_nodes: int | None = None
        self._cached_df_returns = None
        self._cached_dates = None
        self._current_measure_tag: str = "measure"

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

        if not ACE_AVAILABLE:
            self._append_log(
                "Maximal correlation (ACE) unavailable: "
                f"{ACE_IMPORT_ERROR or 'ace_cream not installed'}. "
                "Install a Fortran compiler (gfortran) and run "
                "`uv sync --extra ace` to enable it."
            )

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

        form.addSpacing(4)
        self.btn_evolution_settings = QPushButton("⚙ Evolution Settings")
        self.btn_evolution_settings.setObjectName("SecondaryButton")
        self.btn_evolution_settings.clicked.connect(self._show_evolution_settings)
        self.btn_evolution_settings.setToolTip("Configure network evolution analysis parameters")
        form.addWidget(self.btn_evolution_settings)

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

        self.tabs = QTabWidget()
        self.tabs.setObjectName("ResultTabs")
        self.tabs.setMinimumHeight(300)  # Ensure tabs are always visible

        # --- Network tab ---
        network_page = QFrame()
        network_page.setObjectName("Canvas")
        network_layout = QVBoxLayout(network_page)
        network_layout.setContentsMargins(0, 0, 0, 0)
        self.web = QWebEngineView()
        self.web.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.web.setHtml(self._placeholder_html())
        network_layout.addWidget(self.web)
        self.tabs.addTab(network_page, "Network")

        # --- Degree histogram tab ---
        hist_page = QFrame()
        hist_page.setObjectName("Canvas")
        hist_layout = QVBoxLayout(hist_page)
        hist_layout.setContentsMargins(8, 8, 8, 8)
        self.hist_container = QWidget()
        self.hist_canvas_layout = QVBoxLayout(self.hist_container)
        self.hist_canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.hist_label = QLabel()
        self.hist_label.setObjectName("HistLabel")
        self.hist_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hist_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.hist_label.setMinimumHeight(320)
        self._set_hist_placeholder()
        self.hist_canvas_layout.addWidget(self.hist_label)
        hist_layout.addWidget(self.hist_container)
        self.tabs.addTab(hist_page, "Degree histogram")

        # --- Evolution: Weighted Degree tab ---
        evolution_deg_page = QFrame()
        evolution_deg_page.setObjectName("Canvas")
        evolution_deg_layout = QVBoxLayout(evolution_deg_page)
        evolution_deg_layout.setContentsMargins(8, 8, 8, 8)
        self.evolution_degree_container = QWidget()
        self.evolution_degree_layout = QVBoxLayout(self.evolution_degree_container)
        self.evolution_degree_layout.setContentsMargins(0, 0, 0, 0)
        self.evolution_degree_label = QLabel()
        self.evolution_degree_label.setObjectName("HistLabel")
        self.evolution_degree_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.evolution_degree_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.evolution_degree_label.setMinimumHeight(400)
        self._set_evolution_deg_placeholder()
        self.evolution_degree_layout.addWidget(self.evolution_degree_label)
        evolution_deg_layout.addWidget(self.evolution_degree_container)
        self.tabs.addTab(evolution_deg_page, "Evolution: Degrees")

        # --- Evolution: Centrality tab ---
        evolution_cent_page = QFrame()
        evolution_cent_page.setObjectName("Canvas")
        evolution_cent_layout = QVBoxLayout(evolution_cent_page)
        evolution_cent_layout.setContentsMargins(8, 8, 8, 8)
        self.evolution_centrality_container = QWidget()
        self.evolution_centrality_layout = QVBoxLayout(self.evolution_centrality_container)
        self.evolution_centrality_layout.setContentsMargins(0, 0, 0, 0)
        self.evolution_centrality_label = QLabel()
        self.evolution_centrality_label.setObjectName("HistLabel")
        self.evolution_centrality_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.evolution_centrality_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.evolution_centrality_label.setMinimumHeight(400)
        self._set_evolution_cent_placeholder()
        self.evolution_centrality_layout.addWidget(self.evolution_centrality_label)
        evolution_cent_layout.addWidget(self.evolution_centrality_container)
        self.tabs.addTab(evolution_cent_page, "Evolution: Centrality")

        # --- Evolution: Extended Metrics tab ---
        evolution_ext_page = QFrame()
        evolution_ext_page.setObjectName("Canvas")
        evolution_ext_layout = QVBoxLayout(evolution_ext_page)
        evolution_ext_layout.setContentsMargins(8, 8, 8, 8)
        self.evolution_extended_container = QWidget()
        self.evolution_extended_layout = QVBoxLayout(self.evolution_extended_container)
        self.evolution_extended_layout.setContentsMargins(0, 0, 0, 0)
        self.evolution_extended_label = QLabel()
        self.evolution_extended_label.setObjectName("HistLabel")
        self.evolution_extended_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.evolution_extended_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.evolution_extended_label.setMinimumHeight(400)
        self._set_evolution_extended_placeholder()
        self.evolution_extended_layout.addWidget(self.evolution_extended_label)
        evolution_ext_layout.addWidget(self.evolution_extended_container)
        self.tabs.addTab(evolution_ext_page, "Evolution: Extended Metrics")

        # --- Evolution: Communities tab ---
        evolution_comm_page = QFrame()
        evolution_comm_page.setObjectName("Canvas")
        evolution_comm_layout = QVBoxLayout(evolution_comm_page)
        evolution_comm_layout.setContentsMargins(8, 8, 8, 8)
        self.evolution_community_container = QWidget()
        self.evolution_community_layout = QVBoxLayout(self.evolution_community_container)
        self.evolution_community_layout.setContentsMargins(0, 0, 0, 0)
        self.evolution_community_label = QLabel()
        self.evolution_community_label.setObjectName("HistLabel")
        self.evolution_community_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.evolution_community_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.evolution_community_label.setMinimumHeight(400)
        self._set_evolution_community_placeholder()
        self.evolution_community_layout.addWidget(self.evolution_community_label)
        evolution_comm_layout.addWidget(self.evolution_community_container)
        self.tabs.addTab(evolution_comm_page, "Evolution: Communities")

        layout.addWidget(self.tabs, stretch=3)

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

    def _set_hist_placeholder(self, message: str | None = None) -> None:
        """Set placeholder text for histogram tab."""
        # Clear any existing canvas and close figures
        while self.hist_canvas_layout.count():
            widget = self.hist_canvas_layout.takeAt(0).widget()
            if isinstance(widget, FigureCanvas) and widget.figure:
                plt.close(widget.figure)
            if widget:
                widget.deleteLater()

        # Add placeholder label
        text = message or (
            "Degree histogram will appear here after you build a network."
        )
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #94a3b8; font-size: 14px; background: transparent;")
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.hist_canvas_layout.addWidget(label)

    def _set_hist_building(self) -> None:
        self._set_hist_placeholder(
            "Building… previous histogram cleared. See the process log for progress."
        )

    def _create_threshold_slider(
        self,
        layout: QVBoxLayout,
        fig: Figure,
        canvas: FigureCanvas,
        measure_df: "pl.DataFrame",
        measure_name: str,
    ) -> None:
        """Create a threshold slider for dynamic degree histogram adjustment."""
        from tgraphportfolio.analysis.degree_hist import histogram_title, render_degree_histogram
        from tgraphportfolio.analysis.network import build_corr_nx
        import polars as pl

        # Determine slider range based on measure type
        if measure_name == "distance_correlation":
            min_val, max_val = 0.0, 1.0
        else:  # Pearson, Spearman, ACE
            min_val, max_val = -1.0, 1.0

        # Create slider container
        slider_container = QWidget()
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(4, 0, 4, 0)

        # Label showing current threshold
        lbl_threshold = QLabel("Threshold:")
        lbl_threshold.setStyleSheet("color: #e2e8f0; font-size: 9px;")
        slider_layout.addWidget(lbl_threshold)

        # Slider
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(min_val * 100), int(max_val * 100))
        slider.setValue(int(self.spin_threshold.value() * 100))
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(1)  # Tick every 0.01
        slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                background: #334155;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #38bdf8;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #7dd3fc;
            }
            """
        )
        slider_layout.addWidget(slider, 1)

        # Value display
        lbl_value = QLabel(f"{self.spin_threshold.value():.2f}")
        lbl_value.setStyleSheet("color: #cbd5e1; font-size: 9px; min-width: 35px; text-align: right;")
        slider_layout.addWidget(lbl_value)

        slider_container.setLayout(slider_layout)
        layout.addWidget(slider_container)

        # Store references and connect signal
        def on_slider_changed(value: int) -> None:
            try:
                threshold = value / 100.0
                lbl_value.setText(f"{threshold:.2f}")

                # Rebuild network with new threshold
                new_graph = build_corr_nx(measure_df, independent_threshold=threshold)

                # Get the title from the current figure (approximately)
                hist_title_text = fig.axes[0].get_title() if fig.axes else "Degree Distribution"

                # Render new histogram
                new_fig = render_degree_histogram(new_graph, hist_title_text, threshold)

                # Store data on figure for cursor
                if hasattr(fig, "_histogram_cursor_data"):
                    ax, degrees, bin_edges = fig._histogram_cursor_data
                    new_fig._histogram_cursor_data = (new_fig.axes[0], dict(new_graph.degree()), new_fig.axes[0].patches[0].get_height() if new_fig.axes[0].patches else np.array([]))

                # Update canvas with new figure
                canvas.figure.clear()
                canvas.figure = new_fig
                canvas.draw()

                # Re-attach cursor if needed
                if hasattr(new_fig, "_histogram_cursor_data"):
                    try:
                        from tgraphportfolio.analysis.degree_hist import _HistogramCursor
                        ax, degrees, bin_edges = new_fig._histogram_cursor_data
                        if isinstance(bin_edges, dict):
                            bin_edges = np.linspace(0, max(degrees.values()) + 1, 16)
                        cursor = _HistogramCursor(new_fig.axes[0], degrees, bin_edges, canvas)
                        canvas._cursor = cursor
                    except Exception:
                        pass

            except Exception as e:
                self._append_log(f"Error updating histogram: {str(e)}")

        slider.sliderMoved.connect(on_slider_changed)
        slider.valueChanged.connect(on_slider_changed)

    def _show_histogram_png(self, fig: Figure) -> None:
        """Display degree histogram as interactive matplotlib canvas with threshold slider."""
        try:
            # Clear any existing widgets and close old figures
            while self.hist_canvas_layout.count():
                widget = self.hist_canvas_layout.takeAt(0).widget()
                if isinstance(widget, FigureCanvas) and widget.figure:
                    plt.close(widget.figure)
                if hasattr(widget, 'deleteLater'):
                    widget.deleteLater()

            # Create container for canvas + slider
            container = QWidget()
            container_layout = QVBoxLayout()
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(8)

            # Create and add canvas
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color: transparent;")
            container_layout.addWidget(canvas, 1)
            canvas.draw()

            # Add threshold slider if measure_df is available
            if hasattr(fig, "_measure_df_stored") and fig._measure_df_stored is not None:
                self._create_threshold_slider(
                    container_layout, fig, canvas, fig._measure_df_stored, fig._measure_name
                )

            # Attach cursor if data is available (store reference to prevent garbage collection)
            if hasattr(fig, "_histogram_cursor_data"):
                try:
                    from tgraphportfolio.analysis.degree_hist import _HistogramCursor
                    ax, degrees, bin_edges = fig._histogram_cursor_data
                    cursor = _HistogramCursor(ax, degrees, bin_edges, canvas)
                    canvas._cursor = cursor  # Keep reference to prevent garbage collection
                except Exception as e:
                    self._append_log(f"Cursor error (histogram): {str(e)}")

            container.setLayout(container_layout)
            self.hist_canvas_layout.addWidget(container)
        except Exception as e:
            self._append_log(f"Histogram display error: {str(e)}")

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
            self.btn_evolution_settings,
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
        self._current_measure_tag = measure_short_label(config.measure)
        self.process_log.clear()
        self.lbl_status.setText("Starting…")
        self._append_log("Starting pipeline…")
        # Clear any previously rendered views before the new run.
        self.web.setHtml(self._building_html())
        self._set_hist_building()
        self._set_evolution_building()
        self.tabs.setCurrentIndex(0)

        # Keep strong Python refs — a local worker is GC'd and the thread dies
        # before run() executes (see "QThread: Destroyed while thread is still running").
        self._worker_thread = QThread(self)
        self._worker = PipelineWorker(config, data_cache=self._data_cache)
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
        tag = getattr(self, "_current_measure_tag", "measure")
        self._append_log(
            f"{tag} {bar} {pct:3d}% ({done}/{total})  {desc}",
            replace_last=True,
        )
        self.lbl_status.setText(f"Computing pairs: {desc}")

    def _cleanup_worker(self) -> None:
        self._worker = None
        self._worker_thread = None

    def _on_finished(self, result: object) -> None:
        if not isinstance(result, PipelineResult):
            self._on_failed(f"Unexpected pipeline result type: {type(result)!r}")
            return
        self.progress.setValue(100)
        self.lbl_status.setText("Network ready.")
        self._append_log("Network ready.")
        tmp = tempfile.NamedTemporaryFile(
            prefix="tgraph_", suffix=".html", delete=False
        )
        tmp.write(result.network_html.encode("utf-8"))
        tmp.close()
        if self._temp_html and self._temp_html.exists():
            try:
                self._temp_html.unlink()
            except OSError:
                pass
        self._temp_html = Path(tmp.name)
        self.web.load(QUrl.fromLocalFile(str(self._temp_html.resolve())))
        self._show_histogram_png(result.degree_hist_fig)

        # Store prepared data for evolution worker
        if self._worker is not None:
            self._cached_df_returns = self._worker.df_returns
            self._cached_dates = self._worker.dates

        self._cleanup_worker()

        # Launch evolution analysis
        if self._cached_df_returns is not None and self._cached_dates is not None:
            self._launch_evolution_worker()
        else:
            self._set_busy(False)

    def _on_failed(self, message: str) -> None:
        self.progress.setValue(0)
        self.lbl_status.setText("Failed.")
        self._append_log(f"ERROR: {message}")
        self._set_hist_placeholder("Build failed — histogram unavailable.")
        self._set_evolution_deg_placeholder("Build failed — metrics unavailable.")
        self._set_evolution_cent_placeholder("Build failed — metrics unavailable.")
        self._cleanup_worker()
        self._set_busy(False)
        QMessageBox.critical(self, "Pipeline failed", message)

    # ============================================================================
    # Evolution Analysis Methods
    # ============================================================================

    def _set_evolution_deg_placeholder(self, message: str | None = None) -> None:
        """Set placeholder text for evolution degree tab."""
        # Clear any existing canvas and close figures
        while self.evolution_degree_layout.count():
            widget = self.evolution_degree_layout.takeAt(0).widget()
            if isinstance(widget, FigureCanvas) and widget.figure:
                plt.close(widget.figure)
            if widget:
                widget.deleteLater()

        # Add placeholder label
        text = message or "Evolution metrics will appear here after you build a network."
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #94a3b8; font-size: 14px; background: transparent;")
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.evolution_degree_layout.addWidget(label)

    def _set_evolution_cent_placeholder(self, message: str | None = None) -> None:
        """Set placeholder text for evolution centrality tab."""
        # Clear any existing canvas and close figures
        while self.evolution_centrality_layout.count():
            widget = self.evolution_centrality_layout.takeAt(0).widget()
            if isinstance(widget, FigureCanvas) and widget.figure:
                plt.close(widget.figure)
            if widget:
                widget.deleteLater()

        # Add placeholder label
        text = message or "Evolution metrics will appear here after you build a network."
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #94a3b8; font-size: 14px; background: transparent;")
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.evolution_centrality_layout.addWidget(label)

    def _set_evolution_extended_placeholder(self, message: str | None = None) -> None:
        """Set placeholder text for evolution extended-metrics tab."""
        # Clear any existing canvas and close figures
        while self.evolution_extended_layout.count():
            widget = self.evolution_extended_layout.takeAt(0).widget()
            if isinstance(widget, FigureCanvas) and widget.figure:
                plt.close(widget.figure)
            if widget:
                widget.deleteLater()

        # Add placeholder label
        text = message or "Evolution metrics will appear here after you build a network."
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #94a3b8; font-size: 14px; background: transparent;")
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.evolution_extended_layout.addWidget(label)

    def _set_evolution_community_placeholder(self, message: str | None = None) -> None:
        """Set placeholder text for evolution communities tab."""
        # Clear any existing canvas and close figures
        while self.evolution_community_layout.count():
            widget = self.evolution_community_layout.takeAt(0).widget()
            if isinstance(widget, FigureCanvas) and widget.figure:
                plt.close(widget.figure)
            if widget:
                widget.deleteLater()

        # Add placeholder label
        text = message or "Evolution metrics will appear here after you build a network."
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #94a3b8; font-size: 14px; background: transparent;")
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.evolution_community_layout.addWidget(label)

    def _set_evolution_building(self) -> None:
        """Set building state for evolution tabs."""
        self._set_evolution_deg_placeholder("Computing evolution metrics...")
        self._set_evolution_cent_placeholder("Computing evolution metrics...")
        self._set_evolution_extended_placeholder("Computing evolution metrics...")
        self._set_evolution_community_placeholder("Computing evolution metrics...")

    def _show_evolution_heatmap_png(self, fig: Figure) -> None:
        """Display evolution degree heatmap."""
        try:
            # Clear any existing widgets and close old figures
            while self.evolution_degree_layout.count():
                widget = self.evolution_degree_layout.takeAt(0).widget()
                if isinstance(widget, FigureCanvas) and widget.figure:
                    plt.close(widget.figure)
                if widget:
                    widget.deleteLater()

            # Create and add canvas
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color: transparent;")
            self.evolution_degree_layout.addWidget(canvas)
            canvas.draw()

            # Attach cursor if data is available (store reference to prevent garbage collection)
            if hasattr(fig, "_heatmap_cursor_data"):
                try:
                    from tgraphportfolio.analysis.evolution_viz import _HeatmapCursor
                    ax, matrix, nodes, window_ends_sorted = fig._heatmap_cursor_data
                    cursor = _HeatmapCursor(ax, matrix, nodes, window_ends_sorted, canvas)
                    canvas._cursor = cursor  # Keep reference to prevent garbage collection
                except Exception as e:
                    self._append_log(f"Cursor error (heatmap): {str(e)}")
        except Exception as e:
            self._append_log(f"Heatmap display error: {str(e)}")

    def _show_evolution_centrality_png(self, fig: Figure) -> None:
        """Display evolution centrality trajectories."""
        try:
            # Clear any existing widgets and close old figures
            while self.evolution_centrality_layout.count():
                widget = self.evolution_centrality_layout.takeAt(0).widget()
                if isinstance(widget, FigureCanvas) and widget.figure:
                    plt.close(widget.figure)
                if widget:
                    widget.deleteLater()

            # Create and add canvas
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color: transparent;")
            self.evolution_centrality_layout.addWidget(canvas)
            canvas.draw()

            # Attach cursor if data is available (store reference to prevent garbage collection)
            if hasattr(fig, "_line_cursor_data"):
                try:
                    from tgraphportfolio.analysis.evolution_viz import _LineCursor
                    # _line_cursor_data is now a dict mapping ax -> (ax, line_data, line_objects, original_styles)
                    cursor_data = fig._line_cursor_data
                    if isinstance(cursor_data, dict):
                        # Multiple axes (top/bottom plots)
                        cursors = []
                        for ax, (ax_obj, line_data, line_objects, original_styles) in cursor_data.items():
                            cursor = _LineCursor(ax_obj, line_data, canvas, line_objects, original_styles)
                            cursors.append(cursor)
                        canvas._cursors = cursors  # Keep reference to prevent garbage collection
                    else:
                        # Legacy: single axis (shouldn't happen with new code, but handle gracefully)
                        ax, line_data, line_objects, original_styles = cursor_data
                        cursor = _LineCursor(ax, line_data, canvas, line_objects, original_styles)
                        canvas._cursor = cursor
                except Exception as e:
                    self._append_log(f"Cursor error (centrality): {str(e)}")
        except Exception as e:
            self._append_log(f"Centrality display error: {str(e)}")

    def _show_evolution_extended_png(self, fig: Figure) -> None:
        """Display extended rolling network metrics (faceted grid)."""
        try:
            # Clear any existing widgets and close old figures
            while self.evolution_extended_layout.count():
                widget = self.evolution_extended_layout.takeAt(0).widget()
                if isinstance(widget, FigureCanvas) and widget.figure:
                    plt.close(widget.figure)
                if widget:
                    widget.deleteLater()

            # Create and add canvas
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color: transparent;")
            self.evolution_extended_layout.addWidget(canvas)
            canvas.draw()

            # Attach cursor if data is available (store reference to prevent garbage collection)
            if hasattr(fig, "_extended_cursor_data"):
                try:
                    from tgraphportfolio.analysis.evolution_viz import _MultiPanelCursor
                    panels = fig._extended_cursor_data
                    cursor = _MultiPanelCursor(panels, canvas)
                    canvas._cursor = cursor  # Keep reference to prevent garbage collection
                except Exception as e:
                    self._append_log(f"Cursor error (extended metrics): {str(e)}")
        except Exception as e:
            self._append_log(f"Extended metrics display error: {str(e)}")

    def _show_evolution_community_png(self, fig: Figure) -> None:
        """Display node x window community-membership heatmap."""
        try:
            # Clear any existing widgets and close old figures
            while self.evolution_community_layout.count():
                widget = self.evolution_community_layout.takeAt(0).widget()
                if isinstance(widget, FigureCanvas) and widget.figure:
                    plt.close(widget.figure)
                if widget:
                    widget.deleteLater()

            # Create and add canvas
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color: transparent;")
            self.evolution_community_layout.addWidget(canvas)
            canvas.draw()

            # Attach cursor if data is available (store reference to prevent garbage collection)
            if hasattr(fig, "_community_cursor_data"):
                try:
                    from tgraphportfolio.analysis.evolution_viz import _CommunityHeatmapCursor
                    ax, matrix, nodes, window_ends = fig._community_cursor_data
                    cursor = _CommunityHeatmapCursor(ax, matrix, nodes, window_ends, canvas)
                    canvas._cursor = cursor  # Keep reference to prevent garbage collection
                except Exception as e:
                    self._append_log(f"Cursor error (communities): {str(e)}")
        except Exception as e:
            self._append_log(f"Community heatmap display error: {str(e)}")

    def _show_evolution_settings(self) -> None:
        """Open evolution settings dialog."""
        # Get current threshold from GUI controls
        current_threshold = float(self.spin_threshold.value())

        # Update config with current threshold
        self._evolution_config.independent_threshold = current_threshold

        dialog = EvolutionSettingsDialog(
            self,
            initial_config=self._evolution_config,
            independent_threshold=current_threshold,
            max_nodes=self._current_n_nodes or 20,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._evolution_config = dialog.get_config()
            self._append_log(
                f"Evolution config: window={self._evolution_config.window_size}, "
                f"step={self._evolution_config.step}, "
                f"centrality={self._evolution_config.centrality}, "
                f"threshold={self._evolution_config.independent_threshold:.2f}"
            )

    def _launch_evolution_worker(self) -> None:
        """Launch evolution analysis in background."""
        if self._cached_df_returns is None or self._cached_dates is None:
            self._append_log("ERROR: No prepared data for evolution analysis")
            self._set_busy(False)
            return

        self._set_evolution_building()
        self._append_log("Starting evolution analysis...")

        # Ensure config has current threshold from GUI
        self._evolution_config.independent_threshold = float(self.spin_threshold.value())

        self._evolution_worker_thread = QThread(self)
        self._evolution_worker = EvolutionWorker(
            self._cached_df_returns,
            self._cached_dates,
            self._evolution_config,
            data_cache=self._data_cache,
        )
        self._evolution_worker.moveToThread(self._evolution_worker_thread)
        self._evolution_worker_thread.started.connect(self._evolution_worker.run)
        self._evolution_worker.progress.connect(self._on_evolution_progress)
        self._evolution_worker.status.connect(self._on_evolution_status)
        self._evolution_worker.finished.connect(self._on_evolution_finished)
        self._evolution_worker.failed.connect(self._on_evolution_failed)
        self._evolution_worker.finished.connect(self._evolution_worker_thread.quit)
        self._evolution_worker.failed.connect(self._evolution_worker_thread.quit)
        self._evolution_worker_thread.start()

    def _on_evolution_progress(self, done: int, total: int, desc: str) -> None:
        """Handle evolution progress update."""
        if total <= 0:
            return
        # Log periodically
        if done % max(1, total // 10) == 0 or done in (0, total):
            self._append_log(f"Evolution: {desc}")

    def _on_evolution_status(self, message: str) -> None:
        """Handle evolution status message."""
        self.lbl_status.setText(message)
        self._append_log(message)

    def _on_evolution_finished(self, result: object) -> None:
        """Handle evolution completion."""
        if not isinstance(result, EvolutionResult):
            self._on_evolution_failed(f"Unexpected evolution result type: {type(result)!r}")
            return

        self._append_log("Evolution metrics rendered.")
        self._show_evolution_heatmap_png(result.heatmap_fig)
        self._show_evolution_centrality_png(result.centrality_fig)
        self._show_evolution_extended_png(result.extended_metrics_fig)
        self._show_evolution_community_png(result.community_fig)
        self.lbl_status.setText("Network and evolution analysis complete.")

        self._evolution_worker = None
        self._evolution_worker_thread = None
        self._set_busy(False)

    def _on_evolution_failed(self, message: str) -> None:
        """Handle evolution failure."""
        self._append_log(f"Evolution ERROR: {message}")
        self._set_evolution_deg_placeholder(f"Failed: {message}")
        self._set_evolution_cent_placeholder(f"Failed: {message}")
        self._set_evolution_extended_placeholder(f"Failed: {message}")
        self._set_evolution_community_placeholder(f"Failed: {message}")
        self._evolution_worker = None
        self._evolution_worker_thread = None
        self._set_busy(False)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(2000)
        if self._temp_html and self._temp_html.exists():
            try:
                self._temp_html.unlink()
            except OSError:
                pass
        try:
            self._data_cache.frame_cache.cache_container.close()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)
