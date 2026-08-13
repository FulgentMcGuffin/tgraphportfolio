"""Settings dialog for network evolution analysis configuration."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tgraphportfolio.analysis.evolution import EvolutionConfig
from tgraphportfolio.gui.styles import APP_STYLE, BG_SIDEBAR


class EvolutionSettingsDialog(QDialog):
    """Dialog for configuring temporal network evolution parameters."""

    def __init__(self, parent: QWidget | None = None,
                 initial_config: EvolutionConfig | None = None,
                 independent_threshold: float = 0.33,
                 max_nodes: int | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Evolution Analysis Settings")
        self.setModal(True)
        self.resize(400, 340)
        self.setStyleSheet(APP_STYLE)
        self._force_dark_bg()

        self.initial_config = initial_config or EvolutionConfig(
            independent_threshold=independent_threshold
        )
        self.max_nodes = max_nodes or 20

        self._build_form()

    def _force_dark_bg(self) -> None:
        """Force dark background."""
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(BG_SIDEBAR))
        self.setPalette(palette)

    def _build_form(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Section label
        title = QLabel("Network Evolution Configuration")
        title.setStyleSheet("font-weight: 600; color: #ffffff; font-size: 13px;")
        layout.addWidget(title)

        # Form
        form = QFormLayout()
        form.setSpacing(10)

        # Window size
        self.spin_window_size = QSpinBox()
        self.spin_window_size.setRange(2, 504)
        self.spin_window_size.setValue(self.initial_config.window_size)
        self.spin_window_size.setSuffix(" observations")
        form.addRow("Window size:", self.spin_window_size)

        # Step size
        self.spin_step = QSpinBox()
        self.spin_step.setRange(2, 252)
        self.spin_step.setValue(self.initial_config.step)
        self.spin_step.setSuffix(" observations")
        form.addRow("Step size:", self.spin_step)

        # Min nodes
        self.spin_min_nodes = QSpinBox()
        self.spin_min_nodes.setRange(2, self.max_nodes)
        self.spin_min_nodes.setValue(self.initial_config.min_nodes)
        form.addRow("Min nodes/window:", self.spin_min_nodes)

        # Centrality
        self.cmb_centrality = QComboBox()
        self.cmb_centrality.addItems(["eigenvector", "betweenness", "degree"])
        self.cmb_centrality.setCurrentText(self.initial_config.centrality)
        form.addRow("Centrality measure:", self.cmb_centrality)

        # Top N nodes (for centrality trajectories plot)
        self.spin_n_nodes = QSpinBox()
        self.spin_n_nodes.setRange(1, 20)
        self.spin_n_nodes.setValue(self.initial_config.n_top_nodes)
        form.addRow("Top nodes to plot:", self.spin_n_nodes)

        # Max communities (for per-window ASE+KMeans community detection)
        self.spin_max_communities = QSpinBox()
        self.spin_max_communities.setRange(2, 15)
        self.spin_max_communities.setValue(self.initial_config.max_communities)
        form.addRow("Max communities:", self.spin_max_communities)

        # Independent threshold (read-only, informational)
        lbl_threshold = QLabel(f"{self.initial_config.independent_threshold:.2f}")
        form.addRow("Edge threshold:", lbl_threshold)

        layout.addLayout(form)
        layout.addSpacing(12)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self) -> EvolutionConfig:
        """Return configured EvolutionConfig."""
        return EvolutionConfig(
            window_size=self.spin_window_size.value(),
            step=self.spin_step.value(),
            expanding=False,  # Always rolling for now
            min_nodes=self.spin_min_nodes.value(),
            independent_threshold=self.initial_config.independent_threshold,
            centrality=self.cmb_centrality.currentText(),
            n_top_nodes=self.spin_n_nodes.value(),
            max_communities=self.spin_max_communities.value(),
        )
