import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import faulthandler, signal
faulthandler.enable(all_threads=True)
import os
import sys

from pathlib import Path

from utilities import set_cuda_paths, configure_logging
set_cuda_paths()

from ctypes import windll, byref, sizeof, c_void_p, c_int
from ctypes.wintypes import BOOL, HWND, DWORD

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget,
    QMenuBar, QHBoxLayout, QMessageBox, QDialog, QDialogButtonBox
)
import yaml

from initialize import main as initialize_system
from gui_tabs import create_tabs
from gui_tabs_databases import DatabasesTab
from themes import (
    list_theme_files,
    load_stylesheet,
    ensure_theme_config,
    make_theme_changer,
    MENU_OVERRIDE_QSS
)
from gui_tabs_settings import GuiSettingsTab
from process_manager import get_process_manager

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(800, 560)

        v = QVBoxLayout(self)
        self.settings_tab = GuiSettingsTab()
        v.addWidget(self.settings_tab)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        v.addWidget(buttons)

    def _on_apply(self):
        self.settings_tab.update_all_settings()

    def _on_ok(self):
        self._on_apply()
        self.accept()


class DocQA_GUI(QWidget):
    def __init__(self):
        super().__init__()
        initialize_system()
        self.tab_widget = create_tabs()
        self.databases_tab = self._find_databases_tab()
        self.init_ui()
        self.init_menu()
        self.set_dark_titlebar()

    def _find_databases_tab(self):
        for i in range(self.tab_widget.count()):
            w = self.tab_widget.widget(i)
            if isinstance(w, DatabasesTab):
                return w
        return None

    def set_dark_titlebar(self):
        try:
            DWMWA_USE_IMMERSIVE_DARK_MODE = DWORD(20)
            set_window_attribute = windll.dwmapi.DwmSetWindowAttribute
            hwnd = HWND(int(self.winId()))
            rendering_policy = BOOL(True)
            set_window_attribute(
                hwnd,
                DWMWA_USE_IMMERSIVE_DARK_MODE,
                byref(rendering_policy),
                sizeof(rendering_policy)
            )
            DWMWA_BORDER_COLOR = DWORD(34)
            black_color = c_int(0xFF000000)
            set_window_attribute(
                hwnd,
                DWMWA_BORDER_COLOR,
                byref(black_color),
                sizeof(black_color)
            )
        except Exception:
            pass

    def init_ui(self):
        self.setWindowTitle('VectorDB (Light)')
        self.setGeometry(300, 300, 820, 1000)
        self.setMinimumSize(350, 410)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tab_widget)

    def init_menu(self):
        self.menu_bar = QMenuBar(self)

        self.file_menu = self.menu_bar.addMenu('File')
        settings_action = self.file_menu.addAction('Settings…')
        settings_action.setShortcut('Ctrl+,')
        settings_action.triggered.connect(self.show_settings_dialog)

        self.theme_menu = self.menu_bar.addMenu('Themes')
        for theme in list_theme_files():
            self.theme_menu.addAction(theme).triggered.connect(make_theme_changer(theme))

        layout = self.layout()
        if layout:
            hbox = QHBoxLayout()
            hbox.addWidget(self.menu_bar)
            layout.setMenuBar(self.menu_bar)

    def show_settings_dialog(self):
        dlg = SettingsDialog(self)
        dlg.exec()

    def closeEvent(self, event):
        docs_dir = Path(__file__).parent / 'Docs_for_DB'
        for item in docs_dir.glob('*'):
            if item.is_file():
                try:
                    item.unlink()
                except Exception:
                    pass

        if isinstance(self.tab_widget, QTabWidget):
            for i in range(self.tab_widget.count()):
                tab = self.tab_widget.widget(i)
                if hasattr(tab, 'cleanup') and callable(tab.cleanup):
                    try:
                        tab.cleanup()
                    except Exception:
                        pass

        get_process_manager().cleanup_all(timeout=5.0)

        super().closeEvent(event)


def main():
    from PySide6.QtCore import Qt
    if hasattr(QApplication, 'setHighDpiScaleFactorRoundingPolicy'):
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    configure_logging("INFO")
    app = QApplication(sys.argv)
    theme = ensure_theme_config()
    app.setStyleSheet(load_stylesheet(theme) + MENU_OVERRIDE_QSS)
    ex = DocQA_GUI()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()