import re
import yaml
from pathlib import Path

from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QLineEdit,
    QGridLayout,
    QMessageBox,
    QSizePolicy,
    QCheckBox,
)

from constants import TOOLTIPS
from config import get_config

class ServerSettingsTab(QWidget):
    def __init__(self):
        super(ServerSettingsTab, self).__init__()

        try:
            self.config = get_config()
            self.connection_str = self.config.server.connection_str

            m = re.search(r":(\d{1,5})(?=/)", self.connection_str)
            self.current_port = m.group(1) if m else ""
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Configuration",
                f"An error occurred while loading the configuration: {e}",
            )
            self.server_config = {}
            self.connection_str = ""
            self.current_port = ""
            self.show_thinking = False

        settings_dict = {
            "port": {
                "placeholder": "Port.",
                "validator": QIntValidator(1, 65535),
                "current": self.current_port,
            }
        }

        self.widgets = {}
        layout = QGridLayout()

        port_label = self.create_label("port", settings_dict)
        port_label.setToolTip(TOOLTIPS["PORT"])
        layout.addWidget(port_label, 0, 0)

        port_edit = self.create_edit("port", settings_dict)
        port_edit.setToolTip(TOOLTIPS["PORT"])
        layout.addWidget(port_edit, 0, 1)

        self.setLayout(layout)

    def create_label(self, setting, settings_dict):
        label_text = f"{setting.replace('_', ' ').capitalize()}: {settings_dict[setting]['current']}"
        label = QLabel(label_text)
        self.widgets[setting] = {"label": label}
        return label

    def create_edit(self, setting, settings_dict):
        edit = QLineEdit()
        edit.setPlaceholderText(settings_dict[setting]["placeholder"])
        edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if settings_dict[setting]["validator"]:
            edit.setValidator(settings_dict[setting]["validator"])
        self.widgets[setting]["edit"] = edit
        return edit


    def update_config(self):
        try:
            config = get_config()
            self.connection_str = getattr(getattr(config, "server", object()), "connection_str", "")
            self.server_config = {"connection_str": self.connection_str}
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Configuration",
                f"An error occurred while loading the configuration: {e}",
            )
            return False

        settings_changed = False
        errors = []

        new_port_text = self.widgets["port"]["edit"].text().strip()
        if new_port_text:
            try:
                new_port = int(new_port_text)
                if not (1 <= new_port <= 65535):
                    raise ValueError("Port must be between 1 and 65535.")
            except ValueError:
                errors.append("Port must be an integer between 1 and 65535.")
        else:
            m = re.search(r":(\d{1,5})(?=/)", self.connection_str)
            new_port = m.group(1) if m else ""

        if errors:
            error_message = "\n".join(errors)
            QMessageBox.warning(self, "Invalid Input", f"The following errors occurred:\n{error_message}")
            return False

        if new_port_text and str(new_port) != getattr(self, "current_port", ""):
            m = re.search(r":(\d{1,5})(?=/)", self.connection_str)
            if m:
                new_connection_str = (
                    self.connection_str[: m.start(1)]
                    + str(new_port)
                    + self.connection_str[m.end(1) :]
                )
                config.server.connection_str = new_connection_str
                settings_changed = True
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Connection String",
                    "The existing connection string format is invalid. Unable to update port.",
                )
                return False

        if settings_changed:
            try:
                config.save()

                self.connection_str = getattr(config.server, "connection_str", "")
                m2 = re.search(r":(\d{1,5})(?=/)", self.connection_str)
                self.current_port = m2.group(1) if m2 else ""
                self.widgets["port"]["label"].setText(f"Port: {self.current_port}")
                self.widgets["port"]["edit"].clear()
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Saving Configuration",
                    f"An error occurred while saving the configuration: {e}",
                )
                return False

        return settings_changed