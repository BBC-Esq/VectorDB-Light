# themes.py
from __future__ import annotations

from pathlib import Path
from string import Template
import yaml

from PySide6.QtWidgets import QApplication
from config import get_config

DEFAULT_QSS = r"""
DocQA_GUI {
  background-color: #1e1e1e;
}
QWidget {
  /* border: 2px solid #1e1e1e; */
  border: none;
}
QPushButton {
  background-color: #263238;
  color: #d2d2d2;
  font: 10pt "Segoe UI Historic";
  border-radius: 5px;
  padding: 5px;
  min-width: 60px;
  border: 1px solid transparent;
}
QPushButton:hover {
  background-color: #2F4F4F;
  border: 1px solid #6c757d;
  color: #d2d2d2;
}
QLabel {
  color: #d2d2d2;
}
QComboBox {
  background-color: #263238;
  color: #d2d2d2;
  border: 1px solid #1e1e1e;
  border-radius: 5px;
  padding: 3px;
}
QComboBox:hover,
QComboBox:focus {
  background-color: #2F4F4F;
  color: #d2d2d2;
  border: 1px solid #1e1e1e;
}
QComboBox QAbstractItemView {
  background-color: #161b22;
  color: #d2d2d2;
  border: 1px solid #1e1e1e;
  border-radius: 5px;
}
QComboBox QAbstractItemView::item:hover {
  background-color: #006064;
  color: #d2d2d2;
}
QLineEdit {
  background-color: #1e1e1e;
  color: #a8beb5;
  border: 1px solid transparent;
  border-radius: 5px;
  padding: 3px;
}
QLineEdit:hover,
QLineEdit:focus {
  border: 1px solid #6c757d;
}
QLineEdit::placeholder {
  color: #d67373;
}
QRadioButton {
  color: #d2d2d2;
}
QGroupBox {
  border: 1px solid #161b22;
  border-radius: 5px;
  color: #d2d2d2;
  font-size: 12pt;
  padding: 10px;
}
DownloadModelDialog {
  background-color: #1e1e1e;
}
QFrame {
  background-color: #1e1e1e;
}
QTextEdit[readOnly="true"] {
  background-color: #161b22;
  color: #d2d2d2;
  border: 1px solid #263238;
  border-radius: 5px;
  selection-background-color: #69a9d4;
  selection-color: black;
  font: 14pt "Segoe UI Historic";
}
QTextEdit[readOnly="false"] {
  background-color: #161b22;
  color: #d2d2d2;
  border: 1px solid #263238;
  border-radius: 5px;
  selection-background-color: #69a9d4;
  selection-color: black;
  font: 14pt "Segoe UI Historic";
}
QTabWidget {
  background-color: #1e1e1e;
  border: none;
}
QTabWidget, QTabWidget::pane {
  margin: 0px;
  padding: 0px;
  border: none;
}
QTabBar::tab {
  background-color: #255a7e;
  color: #d2d2d2;
  border-bottom-left-radius: 3px;
  border-bottom-right-radius: 3px;
  margin: 3px;
  padding: 5px 5px;
}
QTabBar::tab:selected {
  background-color: #1e2a88;
  border-bottom: 3px solid #6c757d;
}
QTabBar::tab:hover {
  background-color: #2b3d93;
}
QSplitter::handle {
  background-color: #1B5E20;
  height: 5px;
}
QTreeView {
  color: #d2d2d2;
}
QHeaderView::section {
  background-color: #263238;
  color: #d2d2d2;
  border-radius: 5px;
}
QMenuBar {
  color: #d2d2d2;
}
QMenuBar::item {
  background: transparent;
}
QMenuBar::item:selected {
  background: #4A148C;
}
QCheckBox {
  color: #d2d2d2;
}
QCheckBox::indicator:unchecked:hover,
QCheckBox::indicator:checked:hover {
  border: 1px solid #6c757d;
  border-radius: 5px;
}
QMessageBox {
  background-color: #1e1e1e;
}
QMessageBox QLabel {
  color: #d2d2d2;
}
QMessageBox QPushButton {
  background-color: #255a7e;
  color: #d2d2d2;
  border-radius: 5px;
  padding: 5px;
  border: none;
}
QMessageBox QPushButton:hover,
QMessageBox QPushButton:pressed {
  background-color: #263238;
}
QAbstractItemView {
  background-color: #161b22;
  color: #d2d2d2;
  border: 1px solid #263238;
  border-radius: 5px;
}
QAbstractItemView::item:hover {
  background-color: #006064;
  color: #d2d2d2;
}
QInputDialog {
    background-color: #1e1e1e;
}
QInputDialog QLabel {
    color: #d2d2d2;
}
QInputDialog QComboBox {
    background-color: #263238;
    color: #d2d2d2;
    border: 1px solid #1e1e1e;
    border-radius: 5px;
    padding: 3px;
}
QInputDialog QComboBox:hover {
    background-color: #2F4F4F;
    border: 1px solid #6c757d;
}
QInputDialog QPushButton {
    background-color: #255a7e;
    color: #d2d2d2;
    border-radius: 5px;
    padding: 5px;
    min-width: 60px;
    border: none;
}
QInputDialog QPushButton:hover {
    background-color: #2F4F4F;
    border: 1px solid #6c757d;
}
QDialog {
    background-color: #1e1e1e;
}
QDialog QLabel {
    color: #d2d2d2;
}
QDialog QLineEdit {
    background-color: #161b22;
    color: #d2d2d2;
    border: 1px solid #263238;
    border-radius: 5px;
    padding: 3px;
}
QDialog QLineEdit:hover,
QDialog QLineEdit:focus {
    border: 1px solid #6c757d;
}
QDialog QDialogButtonBox QPushButton {
    background-color: #255a7e;
    color: #d2d2d2;
    border-radius: 5px;
    padding: 5px;
    min-width: 60px;
    border: none;
}
QDialog QDialogButtonBox QPushButton:hover {
    background-color: #2F4F4F;
    border: 1px solid #6c757d;
}
"""

BASE_QSS = Template(r"""
/* Root / base */
DocQA_GUI { background-color: $bg; }
QWidget { color: $fg; border: none; }

/* Buttons */
QPushButton {
  background-color: $btn_bg;
  color: $btn_fg;
  border-radius: 5px;
  padding: 5px;
  min-width: 60px;
  border: 1px solid transparent;
}
QPushButton:hover {
    background-color: $btn_bg_hover;
    border: 1px solid $border_hover;
    color: $btn_hover_fg;
}

/* Labels, radios, checks */
QLabel, QRadioButton, QCheckBox { color: $fg; }
QCheckBox::indicator:unchecked:hover,
QCheckBox::indicator:checked:hover { border: 1px solid $border_hover; border-radius: 5px; }

/* Group boxes / frames */
QGroupBox {
  border: 1px solid $groupbox_border;
  border-radius: 5px;
  color: $fg;
  padding: 10px;
}
QFrame { background-color: $bg; }

/* Combobox + popup */
QComboBox {
  background-color: $combo_bg;
  color: $combo_fg;
  border: 1px solid $combo_border;
  border-radius: 5px;
  padding: 3px;
}
QComboBox:hover, QComboBox:focus {
    border: 1px solid $border_hover;
    color: $combo_hover_fg;
}
QComboBox QAbstractItemView {
  background-color: $combo_popup_bg;
  color: $combo_popup_fg;
  border: 1px solid $combo_popup_border;
  border-radius: 5px;
}
QComboBox QAbstractItemView::item:hover {
    background-color: $combo_item_hover_bg;
    color: $combo_item_hover_fg;
}

/* Line edits */
QLineEdit {
  background-color: $line_bg;
  color: $line_fg;
  border: 1px solid transparent;
  border-radius: 5px;
  padding: 3px;
}
QLineEdit:hover, QLineEdit:focus { border: 1px solid $border_hover; }
QLineEdit::placeholder { color: $line_placeholder; }

/* Text edits */
QTextEdit[readOnly="true"] {
  background-color: $textedit_ro_bg;
  color: $textedit_fg;
  border: 1px solid $textedit_border;
  border-radius: 5px;
  selection-background-color: $selection_bg;
  selection-color: $selection_fg;
}
QTextEdit[readOnly="false"] {
  background-color: $textedit_rw_bg;
  color: $textedit_fg;
  border: 1px solid $textedit_border;
  border-radius: 5px;
  selection-background-color: $selection_bg;
  selection-color: $selection_fg;
}

/* Tabs */
QTabWidget { background-color: $bg; border: none; }
QTabWidget, QTabWidget::pane { margin: 0px; padding: 0px; border: none; }
QTabBar::tab {
  background-color: $tab_bg;
  color: $fg;
  border-bottom-left-radius: 3px;
  border-bottom-right-radius: 3px;
  margin: 3px;
  padding: 5px 5px;
}
QTabBar::tab:selected { background-color: $tab_selected_bg; border-bottom: 3px solid $accent; }
QTabBar::tab:hover { background-color: $tab_hover_bg; }

/* Splitter */
QSplitter::handle { background-color: $splitter_handle_bg; height: 5px; }

/* Tree / lists / tables (generic views) */
QAbstractItemView {
  background-color: $abstract_bg;
  color: $abstract_fg;
  border: 1px solid $abstract_border;
  border-radius: 5px;
}
QAbstractItemView::item:hover { background-color: $abstract_item_hover_bg; color: $abstract_item_hover_fg; }

/* Headers */
QHeaderView::section { background-color: $header_bg; color: $header_fg; border-radius: 5px; }

/* Menu bar */
QMenuBar { color: $menubar_fg; }
QMenuBar::item { background: transparent; }
QMenuBar::item:selected { background: $menu_item_selected_bg; }

/* Message boxes */
QMessageBox { background-color: $msg_bg; }
QMessageBox QLabel { color: $msg_fg; }
QMessageBox QPushButton {
  background-color: $msg_btn_bg; color: $msg_btn_fg;
  border-radius: 5px; padding: 5px; border: none;
}
QMessageBox QPushButton:hover,
QMessageBox QPushButton:pressed { background-color: $msg_btn_hover_bg; }

/* Input dialogs */
QInputDialog { background-color: $inputdlg_bg; }
QInputDialog QLabel { color: $inputdlg_fg; }
QInputDialog QComboBox {
  background-color: $combo_bg; color: $combo_fg;
  border: 1px solid $combo_border; border-radius: 5px; padding: 3px;
}
QInputDialog QComboBox:hover { background-color: $btn_bg_hover; border: 1px solid $border_hover; }
QInputDialog QPushButton {
  background-color: $inputdlg_btn_bg; color: $inputdlg_btn_fg;
  border-radius: 5px; padding: 5px; min-width: 60px; border: none;
}
QInputDialog QPushButton:hover { background-color: $inputdlg_btn_hover_bg; border: 1px solid $border_hover; }

/* Dialogs */
QDialog { background-color: $dialog_bg; }
QDialog QLabel { color: $dialog_fg; }
QDialog QLineEdit {
  background-color: $dialog_line_bg; color: $dialog_line_fg;
  border: 1px solid $dialog_line_border; border-radius: 5px; padding: 3px;
}
QDialog QLineEdit:hover, QDialog QLineEdit:focus { border: 1px solid $border_hover; }
QDialog QDialogButtonBox QPushButton {
  background-color: $dialog_btn_bg; color: $dialog_btn_fg;
  border-radius: 5px; padding: 5px; min-width: 60px; border: none;
}
QDialog QDialogButtonBox QPushButton:hover {
  background-color: $dialog_btn_hover_bg; border: 1px solid $border_hover;
}
""")

MENU_OVERRIDE_QSS = r"""
/* Dark, titlebar-like menus (always on) */
QMenuBar {
  background: #2b2f34;          /* dark grey to match the title area */
  color: #ffffff;
  border-bottom: 1px solid #3a3f45;
}
QMenuBar::item {
  background: transparent;
  padding: 4px 8px;
}
QMenuBar::item:selected {
  background: #3a4047;          /* hover/active */
  color: #ffffff;
}
QMenuBar::item:disabled {
  color: #7f8a95;
}

QMenu {
  background: #2b2f34;
  color: #ffffff;
  border: 1px solid #3a3f45;
}
QMenu::item:selected {
  background: #3a4047;
  color: #ffffff;
}
QMenu::separator {
  height: 1px;
  background: #3a3f45;
  margin: 4px 6px;
}
"""

TYPO_QSS = r"""
/* Global typography — same across all themes */
QPushButton { font: 10pt "Segoe UI Historic"; }
QGroupBox { font-size: 12pt; }
QTextEdit[readOnly="true"] {font: 14pt "Segoe UI Historic"; }
QTextEdit[readOnly="false"] {font: 14pt "Segoe UI Historic"; }
QPlainTextEdit { font: 14pt "Segoe UI Historic"; }
"""

_THEMES_CACHE: dict | None = None

def _load_themes_yaml() -> dict:
    """Read optional themes.yaml next to this module."""
    path = Path(__file__).parent / "themes.yaml"
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        themes = data.get("themes", {})
        return themes if isinstance(themes, dict) else {}
    except Exception:
        return {}

def _get_themes() -> dict:
    global _THEMES_CACHE
    if _THEMES_CACHE is None:
        _THEMES_CACHE = _load_themes_yaml()
    return _THEMES_CACHE

def reload_themes_for_dev() -> None:
    """Optional: call this in dev after editing themes.yaml to hot-reload."""
    global _THEMES_CACHE
    _THEMES_CACHE = _load_themes_yaml()

def list_theme_files() -> list[str]:
    """Return available theme names: 'Default' + YAML-defined themes."""
    names = ["Default"]
    names.extend(sorted(_get_themes().keys()))
    return names

def load_stylesheet(theme_name: str) -> str:
    if theme_name == "Default":
        return DEFAULT_QSS + "\n" + TYPO_QSS

    themes = _get_themes()
    data = themes.get(theme_name)
    if not data:
        return DEFAULT_QSS + "\n" + TYPO_QSS

    if isinstance(data, dict) and "qss" in data:
        qss = data.get("qss") or ""
        return (qss if qss.strip() else DEFAULT_QSS) + "\n" + TYPO_QSS

    vars_dict = data.get("vars", {})
    try:
        rendered = BASE_QSS.substitute(**vars_dict)
    except KeyError as e:
        missing = str(e).strip("'")
        raise KeyError(f"Theme '{theme_name}' is missing key '{missing}' required by BASE_QSS")
    return rendered + "\n" + TYPO_QSS

def ensure_theme_config() -> str:
    """Ensure a theme name is stored in config; default to 'Default'."""
    try:
        config = get_config()
        if not getattr(config.appearance, "theme", None):
            config.appearance.theme = "Default"
            config.save()
            return "Default"
        return config.appearance.theme
    except Exception:
        return "Default"

def update_theme_in_config(new_theme: str) -> None:
    try:
        config = get_config()
        config.appearance.theme = new_theme
        config.save()
    except Exception:
        pass

def make_theme_changer(theme_name: str):
    """Return a callable that applies the theme and persists it."""
    def change_theme():
        app = QApplication.instance()
        app.setStyleSheet(load_stylesheet(theme_name) + MENU_OVERRIDE_QSS)
        update_theme_in_config(theme_name)
    return change_theme
