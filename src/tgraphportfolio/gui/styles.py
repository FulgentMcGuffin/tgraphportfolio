"""Application-wide stylesheet matching the dark data-tool aesthetic."""

# Shared dark surfaces used by QSS and a few programmatic palette fills.
BG_APP = "#0f172a"
BG_SIDEBAR = "#1a1c24"
BG_CONTROL = "#2b3340"
BG_CONTROL_HOVER = "#334155"
BORDER = "#7dd3fc"  # sky blue — matches process-log accent, delineates controls
BORDER_MUTED = "#38bdf8"
TEXT = "#e5e7eb"
TEXT_MUTED = "#94a3b8"
TEXT_LOG = "#7dd3fc"
ACCENT = "#2563eb"
ACCENT_HOVER = "#1d4ed8"

APP_STYLE = f"""
QWidget {{
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 12px;
    color: {TEXT};
    background-color: {BG_APP};
}}
QMainWindow, QWidget#Root {{
    background-color: {BG_APP};
    color: {TEXT};
}}

/* Sidebar containers only — avoid universal descendant border:none. */
QFrame#Sidebar {{
    background-color: {BG_SIDEBAR};
    color: {TEXT};
    border: none;
    border-right: 1px solid #2a2f3a;
}}
QFrame#Sidebar QScrollArea {{
    background-color: {BG_SIDEBAR};
    border: none;
}}
QWidget#SidebarContent {{
    background-color: {BG_SIDEBAR};
    color: {TEXT};
    border: none;
}}

QLabel {{
    background-color: transparent;
    color: {TEXT};
}}
QLabel#Brand {{
    color: #ffffff;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.4px;
    padding-bottom: 2px;
}}
QLabel#SectionTitle {{
    color: {TEXT_MUTED};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    padding-top: 4px;
    padding-bottom: 0px;
}}
QLabel#DbPath, QLabel#StatusLabel {{
    color: {TEXT_MUTED};
    font-size: 11px;
}}
QLabel#LogTitle {{
    color: {TEXT_MUTED};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    background-color: transparent;
}}

QCheckBox {{
    background-color: transparent;
    color: {TEXT};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid {BORDER};
    background-color: {BG_CONTROL};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {BORDER};
}}

QComboBox, QLineEdit, QDateEdit, QDoubleSpinBox, QListWidget {{
    background-color: {BG_CONTROL};
    color: #ffffff;
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 7px;
    selection-background-color: {ACCENT};
    min-height: 22px;
}}
QComboBox:hover, QLineEdit:hover, QDateEdit:hover, QListWidget:hover,
QDoubleSpinBox:hover {{
    border-color: #bae6fd;
    background-color: {BG_CONTROL_HOVER};
}}
QComboBox:focus, QLineEdit:focus, QDateEdit:focus, QDoubleSpinBox:focus,
QListWidget:focus {{
    border: 1px solid #e0f2fe;
}}
QComboBox:disabled, QLineEdit:disabled, QDateEdit:disabled,
QDoubleSpinBox:disabled, QListWidget:disabled {{
    color: #6b7280;
    background-color: #151820;
    border-color: #475569;
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
    background: transparent;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {BORDER};
    width: 0;
    height: 0;
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_CONTROL};
    color: #ffffff;
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    outline: none;
}}
QDateEdit::up-button, QDateEdit::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {BG_CONTROL_HOVER};
    border: none;
    width: 16px;
}}

QListWidget {{
    outline: none;
    padding: 2px;
}}
QListWidget::item {{
    padding: 2px 4px;
    border-radius: 3px;
    color: #ffffff;
}}
QListWidget::item:selected {{
    background-color: {ACCENT};
}}

QPushButton {{
    background-color: {ACCENT};
    color: #ffffff;
    border: 1px solid {BORDER_MUTED};
    border-radius: 16px;
    padding: 7px 14px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {ACCENT_HOVER};
    border-color: {BORDER};
}}
QPushButton:pressed {{
    background-color: #1e40af;
}}
QPushButton:disabled {{
    background-color: #334155;
    color: {TEXT_MUTED};
    border-color: #475569;
}}
QPushButton#SecondaryButton {{
    background-color: {BG_CONTROL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 5px 10px;
}}
QPushButton#SecondaryButton:hover {{
    background-color: {BG_CONTROL_HOVER};
    border-color: #bae6fd;
}}

QProgressBar {{
    background-color: {BG_CONTROL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    text-align: center;
    color: {TEXT};
    height: 14px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 5px;
}}

QFrame#CanvasFrame {{
    background-color: {BG_APP};
    border: none;
}}
QFrame#Canvas {{
    background-color: {BG_APP};
    border: 1px solid #2a2f3a;
    border-radius: 10px;
}}
QPlainTextEdit#ProcessLog {{
    background-color: {BG_SIDEBAR};
    color: {TEXT_LOG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 11px;
    padding: 8px;
}}

QScrollBar:vertical {{
    background: {BG_SIDEBAR};
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #3a4150;
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {BG_SIDEBAR};
    height: 10px;
}}
QScrollBar::handle:horizontal {{
    background: #3a4150;
    border-radius: 5px;
}}
QSplitter::handle {{
    background-color: #2a2f3a;
    width: 1px;
}}
QCalendarWidget QWidget {{
    background-color: {BG_CONTROL};
    color: #ffffff;
}}
QCalendarWidget QToolButton {{
    color: #ffffff;
    background-color: {BG_CONTROL};
}}
QCalendarWidget QAbstractItemView:enabled {{
    color: #ffffff;
    background-color: {BG_CONTROL};
    selection-background-color: {ACCENT};
}}
"""
