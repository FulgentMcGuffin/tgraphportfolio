"""Main window: configuration sidebar + pyvis network canvas + process log."""

from __future__ import annotations

import tempfile
import threading
from datetime import date, datetime
from pathlib import Path

from PySide6.QtCore import QDate, QThread, QUrl, Qt
from PySide6.QtGui import QColor, QPalette, QPixmap, QTextCursor
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from tgraphportfolio.analysis.config import PipelineConfig
from tgraphportfolio.analysis.data_access import (
    column_date_bounds,
    distinct_values,
    eligible_layer_columns,
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
from tgraphportfolio.analysis.mln import MLNConfig
from tgraphportfolio.analysis.multiplex_data import filter_tables
from tgraphportfolio.analysis.multiplex_plotly import build_multiplex_figure
from tgraphportfolio.analysis.pipeline import PipelineResult
from tgraphportfolio.analysis.transforms import available_transforms
from tgraphportfolio.gui.evolution_settings_dialog import EvolutionSettingsDialog
from tgraphportfolio.gui.mln_bridge import MLNBridge, inject_click_bridge
from tgraphportfolio.gui.mln_settings_dialog import MLNSettingsDialog
from tgraphportfolio.gui.styles import APP_STYLE, BG_SIDEBAR
from tgraphportfolio.gui.workers import (
    EvolutionWorker,
    EvolutionResult,
    MLNResult,
    MLNWorker,
    PipelineWorker,
)

# How long Cancel Render waits for a worker to unwind before detaching from it.
# Long enough for a cooperative worker sitting on a progress checkpoint, short
# enough that the button never feels stuck.
CANCEL_GRACE_MS = 250


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
        # Shared with PipelineWorker/EvolutionWorker: setting this raises inside
        # their progress/status callbacks (checked on every pair/window) to unwind
        # a long-running computation almost immediately, instead of the GUI thread
        # blocking on QThread.wait() until the current computation finishes on its own.
        self._cancel_event = threading.Event()
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

        # Edge/connection measure settings
        from tgraphportfolio.gui.edge_settings_dialog import EdgeSettingsConfig

        self._edge_settings = EdgeSettingsConfig()

        # Multi-layer network (MLN) state -- a third worker stage that runs
        # between the pipeline and the evolution analysis.
        self._mln_config = MLNConfig(layer_column="")
        self._mln_worker_thread: QThread | None = None
        self._mln_worker: MLNWorker | None = None
        self._mln_result: MLNResult | None = None
        self._mln_temp_html: Path | None = None
        self._mln_temp_dir: Path | None = None
        self._mln_layer_column: str | None = None
        self._mln_enabled_this_run = False
        self._evolution_enabled_this_run = False
        # Workers detached by Cancel Render, kept alive until their thread
        # actually exits. Dropping the last Python reference to a still-running
        # QThread/worker lets Qt delete the C++ object underneath it and crash.
        self._retired_workers: list[tuple] = []
        self._last_config: PipelineConfig | None = None
        self._mln_row_of: dict[tuple[str, str], int] = {}
        self._mln_bridge = MLNBridge()
        self._mln_channel: QWebChannel | None = None

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
        self.cmb_date.currentTextChanged.connect(self._on_role_column_changed)
        form.addWidget(self.cmb_date)

        form.addWidget(self._section("NODE NAME COLUMN"))
        self.cmb_name = QComboBox()
        # A column claimed as node-name/value can no longer be an MLN layer.
        self.cmb_name.currentTextChanged.connect(self._on_role_column_changed)
        form.addWidget(self.cmb_name)

        form.addWidget(self._section("SERIES VALUE COLUMN"))
        self.cmb_value = QComboBox()
        self.cmb_value.currentTextChanged.connect(self._on_role_column_changed)
        form.addWidget(self.cmb_value)

        filter_body = self._collapsible_section(form, "OPTIONAL FILTER", collapsed=True)
        self.radio_filter_column = QRadioButton("Column value")
        self.radio_filter_where = QRadioButton("WHERE clause")
        self.radio_filter_column.setChecked(True)
        self.filter_mode_group = QButtonGroup(self)
        self.filter_mode_group.addButton(self.radio_filter_column)
        self.filter_mode_group.addButton(self.radio_filter_where)
        self.radio_filter_column.toggled.connect(self._on_filter_mode_changed)
        self.radio_filter_where.toggled.connect(self._on_filter_mode_changed)
        filter_body.addWidget(self.radio_filter_column)
        self.cmb_filter_col = QComboBox()
        self.cmb_filter_col.currentTextChanged.connect(self._load_filter_values)
        filter_body.addWidget(self.cmb_filter_col)
        self.cmb_filter_val = QComboBox()
        self.cmb_filter_val.setEditable(True)
        filter_body.addWidget(self.cmb_filter_val)
        filter_body.addWidget(self.radio_filter_where)
        self.txt_filter_where = QLineEdit()
        self.txt_filter_where.setPlaceholderText(
            'e.g. "EqIndex" = \'DAX30\' AND "Close" > 0'
        )
        self.txt_filter_where.setToolTip(
            "Raw SQL boolean expression inserted after WHERE in the load query."
        )
        filter_body.addWidget(self.txt_filter_where)
        self._on_filter_mode_changed()

        transforms_body = self._collapsible_section(
            form, "TRANSFORMS (ORDERED)", collapsed=True
        )
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
        transforms_body.addWidget(self.lst_transforms)
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
        transforms_body.addLayout(tf_row)

        form.addWidget(self._section("CONNECTION MEASURE"))
        self.cmb_measure = QComboBox()
        for measure_id, label in available_measures():
            self.cmb_measure.addItem(label, measure_id)
        form.addWidget(self.cmb_measure)

        self.btn_edge_settings = QPushButton("⚙ Edge Settings")
        self.btn_edge_settings.setObjectName("SecondaryButton")
        self.btn_edge_settings.clicked.connect(self._show_edge_settings)
        self.btn_edge_settings.setToolTip(
            "Configure measure-specific parameters (e.g., stress regime quantile)"
        )
        form.addWidget(self.btn_edge_settings)

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

        evolution_body = self._collapsible_section(form, "EVOLUTION", collapsed=True)
        self.chk_evolution = QCheckBox("Run Evolution")
        self.chk_evolution.setToolTip(
            "Run the rolling-window evolution analysis after the network is "
            "built. This recomputes the connection measure for every window, so "
            "it is much slower than the single network."
        )
        self.chk_evolution.toggled.connect(self._on_evolution_toggled)
        evolution_body.addWidget(self.chk_evolution)
        self.btn_evolution_settings = QPushButton("⚙ Evolution Settings")
        self.btn_evolution_settings.setObjectName("SecondaryButton")
        self.btn_evolution_settings.clicked.connect(self._show_evolution_settings)
        self.btn_evolution_settings.setToolTip(
            "Configure network evolution analysis parameters"
        )
        evolution_body.addWidget(self.btn_evolution_settings)

        mln_body = self._collapsible_section(form, "MLN", collapsed=True)
        self.chk_mln = QCheckBox("Run MLN")
        self.chk_mln.setToolTip(
            "Build a multi-layer network: one layer per distinct value of the "
            "column below, using the same measure, dates and filter as the "
            "single network."
        )
        self.chk_mln.toggled.connect(self._on_mln_toggled)
        mln_body.addWidget(self.chk_mln)
        self.cmb_mln_layer = QComboBox()  # uneditable by design
        self.cmb_mln_layer.setToolTip(
            "Layer column. Only discrete, low-cardinality columns qualify, and "
            "never the date, node-name or series-value columns."
        )
        mln_body.addWidget(self.cmb_mln_layer)
        self.btn_mln_settings = QPushButton("⚙ MLN Settings")
        self.btn_mln_settings.setObjectName("SecondaryButton")
        self.btn_mln_settings.clicked.connect(self._show_mln_settings)
        self.btn_mln_settings.setToolTip(
            "Configure MLN centrality, community method and Jaccard threshold"
        )
        mln_body.addWidget(self.btn_mln_settings)
        self.cmb_mln_layer.setEnabled(False)  # MLN starts unchecked

        form.addSpacing(8)
        self.btn_run = QPushButton("Build network")
        self.btn_run.clicked.connect(self._run_pipeline)
        form.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        form.addWidget(self.progress)

        self.btn_cancel = QPushButton("✕ Cancel Render")
        self.btn_cancel.setObjectName("CancelButton")
        self.btn_cancel.clicked.connect(self._cancel_render)
        self.btn_cancel.setToolTip(
            "Cancel current analysis and clear all networks/graphs"
        )
        self.btn_cancel.setEnabled(False)
        form.addWidget(self.btn_cancel)

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

        # --- MLN tabs (immediately right of the single-network tabs) ---
        self.tabs.addTab(self._build_mln_page(), "MLN")

        mln_metrics_page = QFrame()
        mln_metrics_page.setObjectName("Canvas")
        mln_metrics_outer = QVBoxLayout(mln_metrics_page)
        mln_metrics_outer.setContentsMargins(8, 8, 8, 8)
        self.mln_metrics_container = QWidget()
        self.mln_metrics_layout = QVBoxLayout(self.mln_metrics_container)
        self.mln_metrics_layout.setContentsMargins(0, 0, 0, 0)
        mln_metrics_outer.addWidget(self.mln_metrics_container)
        self._set_mln_metrics_placeholder()
        self.tabs.addTab(mln_metrics_page, "MLN: Metrics")

        mln_comm_page = QFrame()
        mln_comm_page.setObjectName("Canvas")
        mln_comm_outer = QVBoxLayout(mln_comm_page)
        mln_comm_outer.setContentsMargins(8, 8, 8, 8)
        self.mln_community_container = QWidget()
        self.mln_community_layout = QVBoxLayout(self.mln_community_container)
        self.mln_community_layout.setContentsMargins(0, 0, 0, 0)
        mln_comm_outer.addWidget(self.mln_community_container)
        self._set_mln_community_placeholder()
        self.tabs.addTab(mln_comm_page, "MLN: Community")

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
        self.evolution_centrality_layout = QVBoxLayout(
            self.evolution_centrality_container
        )
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
        self.evolution_community_layout = QVBoxLayout(
            self.evolution_community_container
        )
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

    def _collapsible_section(
        self, parent: QVBoxLayout, title: str, *, collapsed: bool = True
    ) -> QVBoxLayout:
        """Add a disclosure header; return the inner layout (hidden if collapsed)."""
        toggle = QToolButton()
        toggle.setObjectName("CollapseHeader")
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        toggle.setCheckable(True)
        toggle.setAutoRaise(True)
        toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        body = QWidget()
        body.setObjectName("CollapseBody")
        inner = QVBoxLayout(body)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(3)

        def _sync(expanded: bool) -> None:
            toggle.setText(f"{'▾' if expanded else '▸'}  {title}")
            body.setVisible(expanded)

        # Add to the parent layout BEFORE the first setVisible: hiding a widget
        # while it is still parentless marks it as a hidden top-level window,
        # which on some platforms makes it reappear as a floating panel instead
        # of expanding inline.
        parent.addWidget(toggle)
        parent.addWidget(body)
        toggle.toggled.connect(_sync)
        toggle.setChecked(not collapsed)
        _sync(not collapsed)
        return inner

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
        from tgraphportfolio.analysis.degree_hist import (
            histogram_title,
            render_degree_histogram,
        )
        from tgraphportfolio.analysis.network import build_corr_nx
        import polars as pl

        # Determine slider range based on measure type
        # Measures normalized to [0, 1]
        normalized_measures = {
            "distance_correlation",
            "kendall_tau",
            "dtw_distance",
            "shrinkage_correlation",
            "conditional_correlation",
            "mutual_information",
            "chatterjee_xi",
            "maximal_correlation",
        }
        if measure_name in normalized_measures:
            min_val, max_val = 0.0, 1.0
        else:  # Pearson, Spearman (return [-1, 1])
            min_val, max_val = -1.0, 1.0

        # Create slider container
        slider_container = QWidget()
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(4, 0, 4, 0)
        slider_layout.setSpacing(6)

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
        slider.setStyleSheet("""
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
            """)
        slider_layout.addWidget(slider, 1)

        # Value display
        lbl_value = QLabel(f"{self.spin_threshold.value():.2f}")
        lbl_value.setStyleSheet(
            "color: #cbd5e1; font-size: 9px; min-width: 35px; text-align: right;"
        )
        slider_layout.addWidget(lbl_value)

        slider_container.setLayout(slider_layout)
        layout.addWidget(slider_container, 0)  # Add with stretch=0 to keep it compact

        # Store references and connect signal
        def on_slider_changed(value: int) -> None:
            try:
                threshold = value / 100.0
                lbl_value.setText(f"{threshold:.2f}")

                # Rebuild network with new threshold
                new_graph = build_corr_nx(measure_df, independent_threshold=threshold)

                # Extract degree data
                degrees_dict = dict(new_graph.degree())
                deg_list = list(degrees_dict.values())

                if not deg_list:
                    self._append_log("No nodes to plot for updated histogram")
                    return

                # Reuse existing figure and axes to maintain canvas size
                ax = fig.axes[0]
                ax.clear()

                # Recalculate histogram bins
                mean_deg = float(np.mean(deg_list))
                lo, hi = int(min(deg_list)), int(max(deg_list))
                _counts, bin_edges = np.histogram(deg_list, bins=15)

                # Redraw histogram on same axes
                sns.histplot(
                    deg_list,
                    bins=bin_edges,
                    kde=True,
                    color="#38bdf8",
                    edgecolor="#0ea5e9",
                    alpha=0.75,
                    line_kws={"color": "#7dd3fc", "linewidth": 2},
                    ax=ax,
                )

                ax.axvline(
                    mean_deg,
                    color="#a78bfa",
                    linewidth=2.5,
                    linestyle="--",
                    label=f"Mean = {mean_deg:2.0f}",
                )

                ax.set_xlim(max(0, lo - 1), hi + 1)
                ax.xaxis.set_major_locator(
                    MaxNLocator(nbins=8, integer=True, prune=None)
                )
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.minorticks_off()

                ax.set_title(
                    fig.axes[0].get_title(), color="#e2e8f0", fontsize=13, pad=10
                )
                ax.set_xlabel("Number of Connections", color="#cbd5e1", fontsize=10)
                ax.set_ylabel("Density", color="#cbd5e1", fontsize=10)
                ax.tick_params(
                    axis="both", which="major", colors="#cbd5e1", labelsize=10
                )
                ax.grid(
                    True,
                    axis="both",
                    which="major",
                    color="#334155",
                    linewidth=0.8,
                    alpha=0.7,
                )
                ax.set_axisbelow(True)

                for spine in ax.spines.values():
                    spine.set_color("#475569")

                legend = ax.legend(loc="best", fontsize=11, frameon=True)
                legend.get_frame().set_facecolor("#1a1c24")
                legend.get_frame().set_edgecolor("#7dd3fc")
                for text in legend.get_texts():
                    text.set_color("#cbd5e1")

                # Re-add node label annotations for min/max/mode degrees
                try:
                    from tgraphportfolio.analysis.degree_hist import (
                        _annotate_extreme_nodes,
                    )

                    _annotate_extreme_nodes(ax, degrees_dict, lo, hi, bin_edges)
                except Exception as e:
                    self._append_log(
                        f"Warning: Could not annotate extreme nodes: {str(e)}"
                    )

                # Update cursor data
                fig._histogram_cursor_data = (ax, degrees_dict, bin_edges)

                # Redraw canvas without resizing
                canvas.draw_idle()

                # Re-attach cursor
                try:
                    from tgraphportfolio.analysis.degree_hist import _HistogramCursor

                    ax, degrees, bin_edges_data = fig._histogram_cursor_data
                    cursor = _HistogramCursor(ax, degrees, bin_edges_data, canvas)
                    canvas._cursor = cursor
                except Exception as e:
                    self._append_log(f"Cursor error during update: {str(e)}")

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
                if hasattr(widget, "deleteLater"):
                    widget.deleteLater()

            # Create and add canvas directly (maintains original size)
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color: transparent;")
            self.hist_canvas_layout.addWidget(canvas)
            canvas.draw()

            # Add threshold slider if measure_df is available
            if (
                hasattr(fig, "_measure_df_stored")
                and fig._measure_df_stored is not None
            ):
                self._create_threshold_slider(
                    self.hist_canvas_layout,
                    fig,
                    canvas,
                    fig._measure_df_stored,
                    fig._measure_name,
                )

            # Attach cursor if data is available (store reference to prevent garbage collection)
            if hasattr(fig, "_histogram_cursor_data"):
                try:
                    from tgraphportfolio.analysis.degree_hist import _HistogramCursor

                    ax, degrees, bin_edges = fig._histogram_cursor_data
                    cursor = _HistogramCursor(ax, degrees, bin_edges, canvas)
                    canvas._cursor = (
                        cursor  # Keep reference to prevent garbage collection
                    )
                except Exception as e:
                    self._append_log(f"Cursor error (histogram): {str(e)}")
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

        for combo in (
            self.cmb_date,
            self.cmb_name,
            self.cmb_value,
            self.cmb_filter_col,
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(columns)
            combo.blockSignals(False)

        self._guess_roles(columns)
        self._load_filter_values(self.cmb_filter_col.currentText())
        self._reload_mln_columns()
        self._guess_date_range()
        self.btn_run.setEnabled(True)
        self._append_log(f"Selected table {table!r} ({len(columns)} columns).")

    def _guess_roles(self, columns: list[str]) -> None:
        lower = {c.lower(): c for c in columns}

        def pick(
            candidates: list[str], combo: QComboBox, fallback_idx: int = 0
        ) -> None:
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

    def _on_filter_mode_changed(self, *_args) -> None:
        """Enable the active filter mode's widget(s), disable the other's."""
        is_where = self.radio_filter_where.isChecked()
        self.cmb_filter_col.setEnabled(not is_where)
        self.cmb_filter_val.setEnabled(not is_where)
        self.txt_filter_where.setEnabled(is_where)
        if is_where:
            self.txt_filter_where.setFocus()
        else:
            self._load_filter_values(self.cmb_filter_col.currentText())

    def _load_filter_values(self, column: str) -> None:
        if not self._db_path or not column or not self.radio_filter_column.isChecked():
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
        # A leading blank keeps "no filter" expressible. Without it the combo
        # auto-selects the first distinct value, which would silently filter
        # every build on a column the user never opted into -- the filter is
        # meant to be optional, and the radio group has no "off" position.
        self.cmb_filter_val.addItem("")
        self.cmb_filter_val.addItems(values)
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
            self.chk_mln,
            self.btn_mln_settings,
            self.chk_evolution,
            self.radio_filter_column,
            self.radio_filter_where,
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
        self.btn_evolution_settings.setEnabled(
            enabled and self.chk_evolution.isChecked()
        )
        self.cmb_filter_col.setEnabled(enabled and self.radio_filter_column.isChecked())
        self.cmb_filter_val.setEnabled(enabled and self.radio_filter_column.isChecked())
        self.txt_filter_where.setEnabled(
            enabled and self.radio_filter_where.isChecked()
        )
        self.cmb_mln_layer.setEnabled(enabled and self.chk_mln.isChecked())
        # Run only makes sense once a DB is loaded.
        if enabled and self._db_path is None:
            self.btn_run.setEnabled(False)

    def _set_busy(self, busy: bool) -> None:
        """Toggle run/cancel only -- every other control stays usable.

        A run happens on background threads, so the rest of the sidebar can be
        edited while it works (setting up the next build). Only "Build network"
        is blocked, because starting a second run on top of a live one is the
        one genuinely unsafe action.
        """
        self._busy = busy
        self.btn_run.setEnabled(not busy and self._db_path is not None)
        self.btn_cancel.setEnabled(busy)

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

        filter_mode = (
            "where_clause" if self.radio_filter_where.isChecked() else "column_value"
        )
        filter_column = None
        filter_value = None
        where_clause = None
        if filter_mode == "column_value":
            filter_column = self.cmb_filter_col.currentText() or None
            filter_value = self.cmb_filter_val.currentText() or None
        else:
            where_clause = self.txt_filter_where.text().strip() or None

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
            filter_mode=filter_mode,
            filter_column=filter_column,
            filter_value=filter_value,
            where_clause=where_clause,
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

        # Resolve the MLN request before starting anything: a skip must warn the
        # user now (while they are still looking at the screen) and leave the
        # single network + evolution to run normally, rather than aborting the
        # whole build or surfacing minutes later mid-chain.
        mln_column, mln_skip = self._mln_request(config)
        if mln_skip:
            QMessageBox.warning(self, "MLN skipped", mln_skip)
            self._append_log(f"MLN skipped: {mln_skip.splitlines()[0]}")
        self._mln_layer_column = mln_column
        self._mln_enabled_this_run = mln_column is not None
        self._evolution_enabled_this_run = self.chk_evolution.isChecked()
        self._last_config = config

        # A *fresh* event per run. Clearing the shared one would resurrect a
        # worker discarded by a previous cancel, which would then keep running
        # and emit into the new run.
        self._cancel_event = threading.Event()

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
        if self._evolution_enabled_this_run:
            self._set_evolution_building()
        else:
            self._clear_evolution_tabs()
        if self._mln_enabled_this_run:
            self._set_mln_building()
        else:
            self._clear_mln_tabs()
        self.tabs.setCurrentIndex(0)

        # Keep strong Python refs — a local worker is GC'd and the thread dies
        # before run() executes (see "QThread: Destroyed while thread is still running").
        self._worker_thread = QThread(self)
        self._worker = PipelineWorker(
            config,
            data_cache=self._data_cache,
            edge_settings=self._edge_settings.to_dict(),
            cancel_event=self._cancel_event,
        )
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker.cancelled.connect(self._worker_thread.quit)
        self._worker_thread.start()

    def _on_status(self, message: str) -> None:
        if self._is_stale(self._worker):
            return
        self.lbl_status.setText(message)
        self._append_log(message)

    def _on_progress(self, done: int, total: int, desc: str) -> None:
        if self._is_stale(self._worker):
            return
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

    def _is_stale(self, current) -> bool:
        """True when a signal came from a worker we have already detached from.

        Cancel Render stops waiting for threads, so a discarded worker can still
        emit afterwards -- possibly while a *new* run is under way. Acting on
        those signals would overwrite the current run's results, or null its
        thread references (which crashes Qt when the live QThread is later
        garbage-collected).

        Identity comes from ``QObject.sender()`` rather than a bound lambda:
        connecting a lambda gives Qt no receiver object, so it invokes the slot
        DIRECTLY on the worker thread instead of queueing it to the GUI thread,
        and touching widgets from there corrupts the heap. Bound methods keep
        the connection queued, which is required for correctness here.

        ``_retired_workers`` keeps discarded senders alive, so sender() cannot
        dangle. Outside signal delivery it is None, which reads as "current" --
        the right answer for a handler invoked directly.
        """
        sender = self.sender()
        return sender is not None and sender is not current

    def _retire_worker(self, worker, thread) -> None:
        """Hold a detached worker/thread until the thread finishes.

        Cancel Render stops waiting for threads, so without this the last
        Python reference would drop while the thread is still running and Qt
        would delete the underlying C++ objects out from under it.
        """
        if worker is None and thread is None:
            return
        if thread is not None and not thread.isRunning():
            return
        entry = (worker, thread)
        self._retired_workers.append(entry)
        if thread is not None:
            thread.finished.connect(lambda e=entry: self._release_retired(e))

    def _release_retired(self, entry) -> None:
        """Drop a retired worker once its thread has actually exited."""
        try:
            self._retired_workers.remove(entry)
        except ValueError:
            pass

    def _cleanup_worker(self) -> None:
        self._worker = None
        self._worker_thread = None

    def _on_finished(self, result: object) -> None:
        if self._is_stale(self._worker):
            return
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

        # MLN runs before evolution: it needs the same settings, and the user
        # wants its results on screen first.
        if self._mln_enabled_this_run and self._mln_layer_column:
            self._launch_mln_worker()
            return

        self._continue_to_evolution()

    def _continue_to_evolution(self) -> None:
        """Single funnel into the evolution stage.

        Every MLN outcome (done, failed, skipped, disabled) reaches evolution
        through here, so the chain behaves identically in all of them --
        except cancellation, which must stop the chain entirely.
        """
        if self._cancel_event.is_set():
            return
        if not self._evolution_enabled_this_run:
            self._set_busy(False)
            return
        if self._cached_df_returns is not None and self._cached_dates is not None:
            self._launch_evolution_worker()
        else:
            self._set_busy(False)

    def _on_failed(self, message: str) -> None:
        if self._is_stale(self._worker):
            return
        self.progress.setValue(0)
        self.lbl_status.setText("Failed.")
        self._append_log(f"ERROR: {message}")
        self._set_hist_placeholder("Build failed — histogram unavailable.")
        self._set_evolution_deg_placeholder("Build failed — metrics unavailable.")
        self._set_evolution_cent_placeholder("Build failed — metrics unavailable.")
        self._cleanup_worker()
        self._set_busy(False)
        QMessageBox.critical(self, "Pipeline failed", message)

    def _on_cancelled(self) -> None:
        """Pipeline worker unwound after a cancellation request.

        ``_cancel_render`` already performs the full UI cleanup; this handler
        only needs to drop the worker refs. It may fire slightly after
        ``_cancel_render`` returns (the signal is queued cross-thread), so it
        must be safe to run on an already-cleaned-up UI — no popups here,
        unlike ``_on_failed``.
        """
        if self._is_stale(self._worker):
            return
        self._cleanup_worker()

    # ============================================================================
    # Multi-Layer Network (MLN) Methods
    # ============================================================================

    def _build_mln_page(self) -> QWidget:
        """The 'MLN' tab: 3D multiplex view + layer checklist + node table."""
        page = QFrame()
        page.setObjectName("Canvas")
        outer = QHBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self.mln_web = QWebEngineView()
        self.mln_web.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.mln_web.setHtml(self._mln_placeholder_html())
        outer.addWidget(self.mln_web, stretch=4)

        side = QWidget()
        side.setMaximumWidth(240)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(4, 6, 6, 6)
        side_layout.setSpacing(4)

        side_layout.addWidget(self._section("VISIBLE LAYERS"))
        self.lst_mln_layers = QListWidget()
        self.lst_mln_layers.setMaximumHeight(130)
        self.lst_mln_layers.itemChanged.connect(self._on_mln_layer_toggled)
        side_layout.addWidget(self.lst_mln_layers)

        side_layout.addWidget(self._section("NODES"))
        self.tbl_mln_nodes = QTableWidget(0, 2)
        self.tbl_mln_nodes.setHorizontalHeaderLabels(["Node", "Layer"])
        self.tbl_mln_nodes.verticalHeader().setVisible(False)
        self.tbl_mln_nodes.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_mln_nodes.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        side_layout.addWidget(self.tbl_mln_nodes, stretch=1)
        outer.addWidget(side, stretch=1)

        # Click bridge: Plotly node clicks -> highlight the matching table row.
        self._mln_channel = QWebChannel(self.mln_web.page())
        self._mln_channel.registerObject("mlnBridge", self._mln_bridge)
        self.mln_web.page().setWebChannel(self._mln_channel)
        self._mln_bridge.node_clicked.connect(self._on_mln_node_clicked)
        return page

    @staticmethod
    def _mln_placeholder_html(
        message: str = "Enable MLN in the sidebar and build a network.",
    ) -> str:
        return (
            "<!DOCTYPE html><html><body style='font-family:Segoe UI,sans-serif;"
            "display:flex;align-items:center;justify-content:center;height:100%;"
            "margin:0;background:#0f172a;color:#94a3b8'>"
            f"<p>{message}</p></body></html>"
        )

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        """Remove every widget from a canvas layout, closing old figures."""
        while layout.count():
            widget = layout.takeAt(0).widget()
            if isinstance(widget, FigureCanvas) and widget.figure:
                plt.close(widget.figure)
            if widget:
                widget.deleteLater()

    def _mln_placeholder_label(self, message: str) -> QLabel:
        label = QLabel(message)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #94a3b8; font-size: 14px; background: transparent;")
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return label

    def _set_mln_metrics_placeholder(self, message: str | None = None) -> None:
        self._clear_layout(self.mln_metrics_layout)
        self.mln_metrics_layout.addWidget(
            self._mln_placeholder_label(
                message or "MLN metrics will appear here after you build a network."
            )
        )

    def _set_mln_community_placeholder(self, message: str | None = None) -> None:
        self._clear_layout(self.mln_community_layout)
        self.mln_community_layout.addWidget(
            self._mln_placeholder_label(
                message or "MLN communities will appear here after you build a network."
            )
        )

    def _set_mln_building(self) -> None:
        """Put the MLN tabs into their 'computing' state."""
        self._set_mln_metrics_placeholder("Computing multi-layer network...")
        self._set_mln_community_placeholder("Computing multi-layer network...")
        self.mln_web.setHtml(
            self._mln_placeholder_html("Computing multi-layer network...")
        )

    def _clear_mln_tabs(self, message: str | None = None) -> None:
        """Clear all three MLN tabs and drop any cached result."""
        text = message or "MLN is disabled for this build."
        self._set_mln_metrics_placeholder(text)
        self._set_mln_community_placeholder(text)
        self.mln_web.setHtml(self._mln_placeholder_html(text))
        self.lst_mln_layers.blockSignals(True)
        self.lst_mln_layers.clear()
        self.lst_mln_layers.blockSignals(False)
        self.tbl_mln_nodes.setRowCount(0)
        self._mln_row_of = {}
        self._mln_result = None

    def _show_mln_metrics(self, fig: Figure) -> None:
        try:
            self._clear_layout(self.mln_metrics_layout)
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color: transparent;")
            self.mln_metrics_layout.addWidget(canvas)
            canvas.draw()
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"MLN metrics display error: {exc}")

    def _show_mln_community(self, fig: Figure) -> None:
        try:
            self._clear_layout(self.mln_community_layout)
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color: transparent;")
            self.mln_community_layout.addWidget(canvas)
            canvas.draw()
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"MLN community display error: {exc}")

    # ---------------------------------------------------------------- sidebar

    def _on_role_column_changed(self, _text: str) -> None:
        """A date/name/value column changed -- MLN eligibility may have shifted."""
        self._reload_mln_columns()

    def _on_mln_toggled(self, checked: bool) -> None:
        """Enable/clear the layer dropdown to match the MLN checkbox."""
        self.cmb_mln_layer.setEnabled(checked)
        if checked:
            self._reload_mln_columns()
        else:
            self.cmb_mln_layer.clear()

    def _reload_mln_columns(self) -> None:
        """Repopulate the layer dropdown with columns eligible as layer keys."""
        # The date/name/value combos are built before the MLN group (which now
        # sits just above "Build network"), and their currentTextChanged is
        # wired straight here -- so tolerate being called before it exists.
        if getattr(self, "chk_mln", None) is None:
            return
        if not self.chk_mln.isChecked() or not self._db_path:
            return
        table = self.cmb_table.currentText()
        if not table:
            return
        exclude = {
            self.cmb_date.currentText(),
            self.cmb_name.currentText(),
            self.cmb_value.currentText(),
        }
        try:
            columns = eligible_layer_columns(self._db_path, table, exclude=exclude)
        except (
            Exception
        ) as exc:  # noqa: BLE001 -- never break the GUI on a schema quirk
            self._append_log(f"MLN: could not inspect columns: {exc}")
            columns = []

        current = self.cmb_mln_layer.currentText()
        self.cmb_mln_layer.blockSignals(True)
        self.cmb_mln_layer.clear()
        self.cmb_mln_layer.addItems(columns)
        if current in columns:
            self.cmb_mln_layer.setCurrentText(current)
        self.cmb_mln_layer.blockSignals(False)
        if not columns:
            self._append_log(
                "MLN: no eligible layer column in this table (needs a discrete "
                "column with 2-50 distinct values, or under 20 if integer)."
            )

    def _show_mln_settings(self) -> None:
        dialog = MLNSettingsDialog(self, initial_config=self._mln_config)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._mln_config = dialog.get_config()
            self._append_log(
                f"MLN settings: centrality={self._mln_config.centrality}, "
                f"jaccard={self._mln_config.jaccard_threshold:.2f}, "
                f"method={self._mln_config.community_method.value}, "
                f"max_communities={self._mln_config.max_communities}"
            )

    def _mln_request(self, config: PipelineConfig) -> tuple[str | None, str | None]:
        """Resolve the MLN request for this build.

        Returns ``(layer_column, skip_reason)``; ``layer_column`` is None when
        MLN is off or must be skipped.
        """
        if not self.chk_mln.isChecked():
            return None, None
        layer_column = self.cmb_mln_layer.currentText().strip()
        if not layer_column:
            return None, (
                "MLN is enabled but no layer column is selected, so it will be "
                "skipped for this build."
            )
        if (
            config.filter_mode == "column_value"
            and config.filter_column
            and config.filter_value
            and layer_column == config.filter_column
        ):
            return None, (
                f"The MLN layer column ({layer_column!r}) is the same column the "
                f"Optional Filter pins to {config.filter_value!r}. Every row would "
                "fall in a single layer, so the multi-layer network would collapse "
                "to the single network.\n\nMLN will be skipped for this build; the "
                "single network and evolution analysis will run normally.\n\n"
                "Pick a different layer column, or clear the filter."
            )
        return layer_column, None

    # ------------------------------------------------------------- MLN worker

    def _launch_mln_worker(self) -> None:
        """Stage 2 of the chain: build the multi-layer network."""
        if self._last_config is None or not self._mln_layer_column:
            self._continue_to_evolution()
            return

        self._append_log("Starting MLN analysis...")
        mln_config = MLNConfig(
            layer_column=self._mln_layer_column,
            centrality=self._mln_config.centrality,
            jaccard_threshold=self._mln_config.jaccard_threshold,
            community_method=self._mln_config.community_method,
            max_communities=self._mln_config.max_communities,
            min_nodes=self._mln_config.min_nodes,
        )

        self._mln_worker_thread = QThread(self)
        self._mln_worker = MLNWorker(
            self._last_config,
            mln_config,
            edge_settings=self._edge_settings.to_dict(),
            cancel_event=self._cancel_event,
        )
        self._mln_worker.moveToThread(self._mln_worker_thread)
        self._mln_worker_thread.started.connect(self._mln_worker.run)
        self._mln_worker.progress.connect(self._on_mln_progress)
        self._mln_worker.status.connect(self._on_mln_status)
        self._mln_worker.finished.connect(self._on_mln_finished)
        self._mln_worker.failed.connect(self._on_mln_failed)
        self._mln_worker.cancelled.connect(self._on_mln_cancelled)
        self._mln_worker.finished.connect(self._mln_worker_thread.quit)
        self._mln_worker.failed.connect(self._mln_worker_thread.quit)
        self._mln_worker.cancelled.connect(self._mln_worker_thread.quit)
        self._mln_worker_thread.start()

    def _cleanup_mln_worker(self) -> None:
        self._mln_worker = None
        self._mln_worker_thread = None

    def _on_mln_progress(self, done: int, total: int, desc: str) -> None:
        if self._is_stale(self._mln_worker):
            return
        if total <= 0:
            return
        self.progress.setValue(int(100 * done / total))
        self.lbl_status.setText(f"MLN: {desc}")

    def _on_mln_status(self, message: str) -> None:
        if self._is_stale(self._mln_worker):
            return
        self.lbl_status.setText(message)
        self._append_log(message)

    def _on_mln_finished(self, result: object) -> None:
        if self._is_stale(self._mln_worker):
            return
        if not isinstance(result, MLNResult):
            self._on_mln_failed(f"Unexpected MLN result type: {type(result)!r}")
            return
        self._mln_result = result
        self._populate_mln_layer_list(result)
        self._populate_mln_table(result)
        self._render_mln_view()
        self._show_mln_metrics(result.metrics_fig)
        self._show_mln_community(result.community_fig)
        self._append_log("MLN rendered.")
        self._cleanup_mln_worker()
        self._continue_to_evolution()

    def _on_mln_failed(self, message: str) -> None:
        if self._is_stale(self._mln_worker):
            return
        self._append_log(f"MLN ERROR: {message}")
        self._clear_mln_tabs(f"MLN failed: {message}")
        self._cleanup_mln_worker()
        # Evolution is independent of the MLN -- keep the chain going.
        self._continue_to_evolution()

    def _on_mln_cancelled(self) -> None:
        """MLN worker unwound after a cancellation request.

        ``_cancel_render`` performs the UI cleanup and the chain must stop here,
        so this only drops the worker refs (see ``_on_cancelled``).
        """
        if self._is_stale(self._mln_worker):
            return
        self._cleanup_mln_worker()

    # ------------------------------------------------------- MLN interactivity

    def _populate_mln_layer_list(self, result: MLNResult) -> None:
        """Fill the visible-layers checklist (all layers checked initially)."""
        self.lst_mln_layers.blockSignals(True)
        self.lst_mln_layers.clear()
        for value in result.layer_values:
            item = QListWidgetItem(value)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.lst_mln_layers.addItem(item)
        self.lst_mln_layers.blockSignals(False)

    def _selected_mln_layers(self) -> list[str]:
        return [
            self.lst_mln_layers.item(i).text()
            for i in range(self.lst_mln_layers.count())
            if self.lst_mln_layers.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _populate_mln_table(self, result: MLNResult) -> None:
        """Fill the node table and the (node, layer) -> row index for O(1) lookup."""
        rows = result.nodes.select(["issuer", "term"]).rows()
        # Suspend repaints while filling: every setItem would otherwise schedule
        # a view update, which is the bulk of the cost for a few thousand rows.
        self.tbl_mln_nodes.setUpdatesEnabled(False)
        try:
            self.tbl_mln_nodes.setRowCount(len(rows))
            self.tbl_mln_nodes.setHorizontalHeaderLabels(
                [result.node_column, result.layer_column]
            )
            self._mln_row_of = {}
            for row_idx, (node, layer) in enumerate(rows):
                self.tbl_mln_nodes.setItem(row_idx, 0, QTableWidgetItem(str(node)))
                self.tbl_mln_nodes.setItem(row_idx, 1, QTableWidgetItem(str(layer)))
                self._mln_row_of[(str(node), str(layer))] = row_idx
            self.tbl_mln_nodes.resizeColumnsToContents()
        finally:
            self.tbl_mln_nodes.setUpdatesEnabled(True)

    def _on_mln_layer_toggled(self, _item) -> None:
        """Re-render the 3D view for the new layer selection.

        Cheap enough for the GUI thread: the multiplex tables are cached on the
        result, so this only re-filters and redraws -- nothing is recomputed.
        """
        self._render_mln_view()

    def _render_mln_view(self) -> None:
        """Draw the cached multiplex for the currently checked layers."""
        result = self._mln_result
        if result is None:
            return
        visible = self._selected_mln_layers()
        if not visible:
            self.mln_web.setHtml(self._mln_placeholder_html("No layers selected."))
            return
        try:
            nodes, intra, inter = filter_tables(
                result.nodes, result.intra, result.inter, visible
            )
            all_nodes = sorted(result.nodes.get_column("issuer").unique().to_list())
            fig = build_multiplex_figure(
                nodes,
                intra,
                inter,
                visible,
                all_issuers=all_nodes,
                title=(
                    f"{result.node_column} x {result.layer_column} "
                    "multi-layer network"
                ),
                layer_label=result.layer_column,
            )
            # 'directory' keeps each page ~50KB by referencing a plotly.min.js
            # written once alongside it; inlining would rewrite 4MB on every
            # layer toggle.
            html = inject_click_bridge(
                fig.to_html(full_html=True, include_plotlyjs="directory")
            )
            self._write_mln_html(html)
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"MLN view render error: {exc}")

    def _mln_asset_dir(self) -> Path:
        """Temp dir holding the MLN page and its one-time plotly.min.js sidecar."""
        if self._mln_temp_dir is None:
            self._mln_temp_dir = Path(tempfile.mkdtemp(prefix="tgraph_mln_"))
        js_path = self._mln_temp_dir / "plotly.min.js"
        if not js_path.exists():
            from plotly.offline import get_plotlyjs

            js_path.write_text(get_plotlyjs(), encoding="utf-8")
        return self._mln_temp_dir

    def _write_mln_html(self, html: str) -> None:
        """Write the MLN page next to its JS sidecar and load it."""
        directory = self._mln_asset_dir()
        handle = tempfile.NamedTemporaryFile(
            prefix="view_", suffix=".html", dir=str(directory), delete=False
        )
        handle.write(html.encode("utf-8"))
        handle.close()
        old = self._mln_temp_html
        self._mln_temp_html = Path(handle.name)
        self.mln_web.load(QUrl.fromLocalFile(str(self._mln_temp_html.resolve())))
        if old and old.exists():
            try:
                old.unlink()
            except OSError:
                pass

    def _on_mln_node_clicked(self, node: str, layer: str) -> None:
        """Select and scroll to the clicked node's row in the side table."""
        row = self._mln_row_of.get((node, layer))
        if row is None:
            return
        self.tbl_mln_nodes.selectRow(row)
        item = self.tbl_mln_nodes.item(row, 0)
        if item is not None:
            self.tbl_mln_nodes.scrollToItem(item)
        self.lbl_status.setText(f"MLN: selected {node} in {layer}")

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
        text = (
            message or "Evolution metrics will appear here after you build a network."
        )
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
        text = (
            message or "Evolution metrics will appear here after you build a network."
        )
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
        text = (
            message or "Evolution metrics will appear here after you build a network."
        )
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
        text = (
            message or "Evolution metrics will appear here after you build a network."
        )
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #94a3b8; font-size: 14px; background: transparent;")
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.evolution_community_layout.addWidget(label)

    def _on_evolution_toggled(self, checked: bool) -> None:
        """Enable/disable the evolution settings button with its checkbox."""
        self.btn_evolution_settings.setEnabled(checked)

    def _clear_evolution_tabs(self, message: str | None = None) -> None:
        """Blank all four evolution canvases (evolution disabled or cancelled)."""
        text = message or "Evolution is disabled for this build."
        self._set_evolution_deg_placeholder(text)
        self._set_evolution_cent_placeholder(text)
        self._set_evolution_extended_placeholder(text)
        self._set_evolution_community_placeholder(text)

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
                    cursor = _HeatmapCursor(
                        ax, matrix, nodes, window_ends_sorted, canvas
                    )
                    canvas._cursor = (
                        cursor  # Keep reference to prevent garbage collection
                    )
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
                        for ax, (
                            ax_obj,
                            line_data,
                            line_objects,
                            original_styles,
                        ) in cursor_data.items():
                            cursor = _LineCursor(
                                ax_obj, line_data, canvas, line_objects, original_styles
                            )
                            cursors.append(cursor)
                        canvas._cursors = (
                            cursors  # Keep reference to prevent garbage collection
                        )
                    else:
                        # Legacy: single axis (shouldn't happen with new code, but handle gracefully)
                        ax, line_data, line_objects, original_styles = cursor_data
                        cursor = _LineCursor(
                            ax, line_data, canvas, line_objects, original_styles
                        )
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
                    canvas._cursor = (
                        cursor  # Keep reference to prevent garbage collection
                    )
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
                    from tgraphportfolio.analysis.evolution_viz import (
                        _CommunityHeatmapCursor,
                    )

                    ax, matrix, nodes, window_ends = fig._community_cursor_data
                    cursor = _CommunityHeatmapCursor(
                        ax, matrix, nodes, window_ends, canvas
                    )
                    canvas._cursor = (
                        cursor  # Keep reference to prevent garbage collection
                    )
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

    def _show_edge_settings(self) -> None:
        """Open edge/connection measure settings dialog."""
        from tgraphportfolio.gui.edge_settings_dialog import EdgeSettingsDialog

        dialog = EdgeSettingsDialog(self, initial_config=self._edge_settings)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._edge_settings = dialog.get_config()
            self._append_log(
                f"Edge settings: conditional_quantile={self._edge_settings.conditional_quantile:.2f}"
            )

    def _cancel_render(self) -> None:
        """Cancel current analysis and clear all visualizations.

        Setting ``_cancel_event`` first is what makes this non-blocking: it's
        checked inside the worker's progress/status callbacks (invoked on
        every pair/window), so the worker unwinds within one iteration rather
        than running to completion. ``QThread.quit()`` alone cannot interrupt
        a synchronous computation already in progress — it only requests the
        thread's event loop to exit once control returns to it.
        """
        self._append_log("Cancelling render...")
        self._cancel_event.set()

        # Give each worker a brief chance to unwind, then carry on regardless.
        # Workers only notice cancellation inside their progress callbacks, so a
        # thread that is momentarily between checkpoints would otherwise stall
        # the GUI here for the whole timeout. Detaching instead keeps the button
        # instant; the discarded thread exits on its next checkpoint and its
        # signals are ignored (see _is_stale).
        for thread, label in (
            (self._worker_thread, "Pipeline"),
            (self._mln_worker_thread, "MLN"),
            (self._evolution_worker_thread, "Evolution"),
        ):
            if thread is None or not thread.isRunning():
                continue
            thread.quit()
            if thread.wait(CANCEL_GRACE_MS):
                self._append_log(f"{label} cancelled.")
            else:
                self._append_log(
                    f"{label} is finishing in the background; its results will "
                    "be discarded."
                )

        # Detach from the workers unconditionally: from here on they are stale.
        self._retire_worker(self._worker, self._worker_thread)
        self._retire_worker(self._mln_worker, self._mln_worker_thread)
        self._retire_worker(self._evolution_worker, self._evolution_worker_thread)
        self._cleanup_worker()
        self._cleanup_mln_worker()
        self._evolution_worker = None
        self._evolution_worker_thread = None

        # Clear all visualizations
        # Clear network canvas
        self.web.setHtml(self._placeholder_html())
        self._clear_mln_tabs("Render cancelled.")
        self._clear_evolution_tabs("Render cancelled.")

        # Clear histogram canvas
        while self.hist_canvas_layout.count():
            widget = self.hist_canvas_layout.takeAt(0).widget()
            if hasattr(widget, "figure") and widget.figure:
                plt.close(widget.figure)
            if hasattr(widget, "deleteLater"):
                widget.deleteLater()

        # Clear tabs
        if hasattr(self, "tabs_evolution"):
            for i in range(self.tabs_evolution.count()):
                tab = self.tabs_evolution.widget(i)
                if hasattr(tab, "canvas") and tab.canvas:
                    if hasattr(tab.canvas, "figure"):
                        plt.close(tab.canvas.figure)

        # Reset UI
        self.progress.setValue(0)
        self.lbl_status.setText("Render cancelled. Polars data retained.")
        self._set_busy(False)
        self._append_log("All networks and graphs cleared.")

    def _launch_evolution_worker(self) -> None:
        """Launch evolution analysis in background."""
        if self._cached_df_returns is None or self._cached_dates is None:
            self._append_log("ERROR: No prepared data for evolution analysis")
            self._set_busy(False)
            return

        self._set_evolution_building()
        self._append_log("Starting evolution analysis...")

        # Ensure config has current threshold from GUI
        self._evolution_config.independent_threshold = float(
            self.spin_threshold.value()
        )
        # Evolution uses the same connection measure as the Network and MLN
        # tabs, rather than silently always using distance correlation.
        if self._last_config is not None:
            self._evolution_config.measure = self._last_config.measure
        self._evolution_config.edge_settings = self._edge_settings.to_dict()

        self._evolution_worker_thread = QThread(self)
        self._evolution_worker = EvolutionWorker(
            self._cached_df_returns,
            self._cached_dates,
            self._evolution_config,
            data_cache=self._data_cache,
            cancel_event=self._cancel_event,
        )
        self._evolution_worker.moveToThread(self._evolution_worker_thread)
        self._evolution_worker_thread.started.connect(self._evolution_worker.run)
        self._evolution_worker.progress.connect(self._on_evolution_progress)
        self._evolution_worker.status.connect(self._on_evolution_status)
        self._evolution_worker.finished.connect(self._on_evolution_finished)
        self._evolution_worker.failed.connect(self._on_evolution_failed)
        self._evolution_worker.cancelled.connect(self._on_evolution_cancelled)
        self._evolution_worker.finished.connect(self._evolution_worker_thread.quit)
        self._evolution_worker.failed.connect(self._evolution_worker_thread.quit)
        self._evolution_worker.cancelled.connect(self._evolution_worker_thread.quit)
        self._evolution_worker_thread.start()

    def _on_evolution_progress(self, done: int, total: int, desc: str) -> None:
        """Handle evolution progress update."""
        if self._is_stale(self._evolution_worker):
            return
        if total <= 0:
            return
        # Log periodically
        if done % max(1, total // 10) == 0 or done in (0, total):
            self._append_log(f"Evolution: {desc}")

    def _on_evolution_status(self, message: str) -> None:
        """Handle evolution status message."""
        if self._is_stale(self._evolution_worker):
            return
        self.lbl_status.setText(message)
        self._append_log(message)

    def _on_evolution_finished(self, result: object) -> None:
        """Handle evolution completion."""
        if self._is_stale(self._evolution_worker):
            return
        if not isinstance(result, EvolutionResult):
            self._on_evolution_failed(
                f"Unexpected evolution result type: {type(result)!r}"
            )
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
        if self._is_stale(self._evolution_worker):
            return
        self._append_log(f"Evolution ERROR: {message}")
        self._set_evolution_deg_placeholder(f"Failed: {message}")
        self._set_evolution_cent_placeholder(f"Failed: {message}")
        self._set_evolution_extended_placeholder(f"Failed: {message}")
        self._set_evolution_community_placeholder(f"Failed: {message}")
        self._evolution_worker = None
        self._evolution_worker_thread = None
        self._set_busy(False)

    def _on_evolution_cancelled(self) -> None:
        """Evolution worker unwound after a cancellation request.

        ``_cancel_render`` already performs the full UI cleanup; this handler
        only needs to drop the worker refs (see ``_on_cancelled`` for why no
        popup/UI work happens here).
        """
        if self._is_stale(self._evolution_worker):
            return
        self._evolution_worker = None
        self._evolution_worker_thread = None

    def closeEvent(self, event) -> None:  # noqa: N802
        # Same fix as _cancel_render: set the flag before quit()/wait() so a
        # worker mid-computation unwinds instead of the 2s wait timing out.
        self._cancel_event.set()
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(2000)
        if self._mln_worker_thread is not None and self._mln_worker_thread.isRunning():
            self._mln_worker_thread.quit()
            self._mln_worker_thread.wait(2000)
        if (
            self._evolution_worker_thread is not None
            and self._evolution_worker_thread.isRunning()
        ):
            self._evolution_worker_thread.quit()
            self._evolution_worker_thread.wait(2000)
        if self._mln_temp_dir and self._mln_temp_dir.exists():
            import shutil

            shutil.rmtree(self._mln_temp_dir, ignore_errors=True)
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
