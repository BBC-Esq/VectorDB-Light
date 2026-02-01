# gui_tabs_settings.py
import logging
from functools import partial
from PySide6.QtWidgets import (
    QVBoxLayout,
    QGroupBox,
    QPushButton,
    QHBoxLayout,
    QWidget,
    QMessageBox,
)
from gui_tabs_settings_server import ServerSettingsTab
from gui_tabs_settings_database_create import ChunkSettingsTab
from gui_tabs_settings_database_query import DatabaseSettingsTab
from gui_file_credentials import manage_credentials

def update_all_configs(configs):
    updated = False
    for config in configs.values():
        updated = config.update_config() or updated

    if updated:
        logging.info("config.yaml file updated")
        QMessageBox.information(None, "Settings Updated", "One or more settings have been updated.")

def adjust_stretch(groups, layout):
    for group, factor in groups.items():
        layout.setStretchFactor(group, factor if group.isChecked() else 0)

class IntegrationsSettings(QWidget):
    def __init__(self):
        super().__init__()
        v = QHBoxLayout()
        self.btn_hf = QPushButton("Manage Hugging Face Access Token")
        self.btn_hf.clicked.connect(lambda: manage_credentials(self, 'hf'))
        v.addWidget(self.btn_hf)

        self.btn_openai = QPushButton("Manage OpenAI API Key")
        self.btn_openai.clicked.connect(lambda: manage_credentials(self, 'openai'))
        v.addWidget(self.btn_openai)

        v.addStretch(1)
        self.setLayout(v)

    def update_config(self):
        return False


class GuiSettingsTab(QWidget):
    def __init__(self):
        super(GuiSettingsTab, self).__init__()
        self.layout = QVBoxLayout()
        classes = {
            "LM Studio Server": (ServerSettingsTab, 2),
            "Database Query": (DatabaseSettingsTab, 4),
            "Database Creation": (ChunkSettingsTab, 3),
            "Integrations": (IntegrationsSettings, 1),
        }
        self.groups = {}
        self.configs = {}

        for title, (TabClass, stretch) in classes.items():
            settings = TabClass()
            group = QGroupBox(title)
            layout = QVBoxLayout()
            layout.addWidget(settings)
            group.setLayout(layout)
            group.setCheckable(True)
            group.setChecked(True)
            self.groups[group] = stretch
            self.configs[title] = settings
            self.layout.addWidget(group, stretch)
            group.toggled.connect(partial(self.toggle_group, group))

        self.setLayout(self.layout)
        adjust_stretch(self.groups, self.layout)

    def toggle_group(self, group, checked):
        if group.title() in self.configs:
            self.configs[group.title()].setVisible(checked)
        adjust_stretch(self.groups, self.layout)

    def update_all_settings(self):
        update_all_configs(self.configs)
