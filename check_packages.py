import importlib.util
import importlib.metadata
import sys
import urllib.request
import json
import subprocess
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QLabel, QMessageBox, QProgressBar, 
    QMenu, QFileDialog, QDialog, QHeaderView, QScrollArea,
    QCheckBox
)
from PySide6.QtCore import Qt, QObject, QThread, Signal, QPoint

from packaging import version

class OutdatedPackagesWorker(QObject):
    finished = Signal(list)
    error = Signal(str)
    def run(self):
        try:
            installed_packages = list(importlib.metadata.distributions())
            outdated_packages = []
            for dist in installed_packages:
                name = dist.metadata['Name']
                version_installed = dist.version
                latest_version = self.get_latest_version(name)
                if latest_version and version.parse(latest_version) > version.parse(version_installed):
                    outdated_packages.append((name, version_installed, latest_version))
            self.finished.emit(outdated_packages)
        except Exception as e:
            self.error.emit(str(e))
    def get_latest_version(self, package_name):
        url = f"https://pypi.org/pypi/{package_name}/json"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status != 200:
                    raise Exception(f"PyPI returned status code {response.status}")
                return json.load(response)['info']['version']
        except urllib.error.URLError as e: 
            raise Exception(f"Network error: {str(e)}")
        except TimeoutError:
            raise Exception("Connection timed out")
        except json.JSONDecodeError:
            raise Exception("Invalid response from PyPI")
        except Exception as e:
            raise Exception(f"Error fetching version: {str(e)}")

class VersionsWorker(QObject):
    finished = Signal(list)
    error = Signal(str)
    def __init__(self, package_name):
        super().__init__()
        self.package_name = package_name
    def run(self):
        try:
            versions = self.get_all_versions(self.package_name)
            self.finished.emit(versions)
        except Exception as e:
            self.error.emit(str(e))
    def get_all_versions(self, package_name):
        url = f"https://pypi.org/pypi/{package_name}/json"
        versions = []
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status != 200:
                    raise Exception(f"PyPI returned status code {response.status}")
                data = json.load(response)
                for ver, release_info in data['releases'].items():
                    if release_info:
                        release_date = release_info[0].get('upload_time', 'N/A')
                        versions.append((ver, release_date))
                versions = sorted(versions, key=lambda v: version.parse(v[0]))
        except urllib.error.URLError as e:
            raise Exception(f"Network error: {str(e)}")
        except TimeoutError:
            raise Exception("Connection timed out")
        except json.JSONDecodeError:
            raise Exception("Invalid response from PyPI")
        except Exception as e:
            raise Exception(f"Error fetching versions: {str(e)}")
        return versions

class PipWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    def __init__(self, package_name, selected_version):
        super().__init__()
        self.package_name = package_name
        self.selected_version = selected_version
    def run(self):
        try:
            command = [sys.executable, "-m", "pip", "install", f"{self.package_name}=={self.selected_version}", "--no-deps"]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                self.finished.emit(stdout)
            else:
                self.error.emit(stderr)
        except Exception as e:
            self.error.emit(str(e))

class PipCheckWorker(QObject):
    finished = Signal()
    def run(self):
        try:
            command = [sys.executable, "-m", "pip", "check"]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

            print("\n=== Pip Check Results ===")
            if stdout.strip():
                print(stdout)
            else:
                print("No broken requirements found.")
            if stderr:
                print("Errors:")
                print(stderr)
            print("========================\n")

            self.finished.emit()
        except Exception as e:
            print(f"\nError running pip check: {str(e)}\n")
            self.finished.emit()

class LatestVersionWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    def __init__(self, package_name):
        super().__init__()
        self.package_name = package_name
    def run(self):
        try:
            latest_version = self.get_latest_version(self.package_name)
            self.finished.emit(latest_version)
        except Exception as e:
            self.error.emit(str(e))
    def get_latest_version(self, package_name):
        url = f"https://pypi.org/pypi/{package_name}/json"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status != 200:
                    raise Exception(f"PyPI returned status code {response.status}")
                return json.load(response)['info']['version']
        except urllib.error.URLError as e:
            raise Exception(f"Network error: {str(e)}")
        except TimeoutError:
            raise Exception("Connection timed out")
        except json.JSONDecodeError:
            raise Exception("Invalid response from PyPI")
        except Exception as e:
            raise Exception(f"Error fetching version: {str(e)}")


class CommandWorker(QObject):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, command):
        super().__init__()
        self.command = command

    def run(self):
        try:
            command_parts = self.command.strip().split()
            if not command_parts:
                self.error.emit("No command provided")
                return

            if command_parts[0].lower() == "pip":
                command_parts = [sys.executable, "-m"] + command_parts

            process = subprocess.Popen(
                command_parts, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                output = stdout if stdout else "Command executed successfully"
                self.finished.emit(output)
            else:
                error_msg = stderr if stderr else f"Command failed with return code {process.returncode}"
                self.error.emit(error_msg)
        except Exception as e:
            self.error.emit(str(e))


class CompareDependenciesDialog(QDialog):
    def __init__(self, parent, package_name, current_version, latest_version, current_deps, latest_deps):
        super().__init__(parent)
        self.setWindowTitle(f"Dependency Comparison - {package_name}")
        self.setMinimumWidth(600)
        self.setFixedHeight(600)
        self.package_name = package_name
        self.current_version = current_version
        self.latest_version = latest_version
        self.current_deps_full = current_deps
        self.latest_deps_full = latest_deps
        self.hide_extras = True

        self.main_layout = QVBoxLayout(self)

        version_layout = QVBoxLayout()
        version_layout.addWidget(QLabel(f"<b>Current Version:</b> {self.current_version}"))
        version_layout.addWidget(QLabel(f"<b>Latest Version:</b> {self.latest_version}"))
        version_layout.addWidget(QLabel("<b>Dependencies:</b>"))
        self.main_layout.addLayout(version_layout)

        self.hide_extras_checkbox = QCheckBox("Hide Extra Dependencies")
        self.hide_extras_checkbox.stateChanged.connect(self.update_display)
        self.main_layout.addWidget(self.hide_extras_checkbox)

        self.deps_layout = QHBoxLayout()
        self.main_layout.addLayout(self.deps_layout)

        self.current_scroll_area = QScrollArea()
        self.current_scroll_area.setWidgetResizable(True)
        self.current_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.current_widget = QWidget()
        self.current_layout = QVBoxLayout(self.current_widget)
        self.current_label = QLabel()
        self.current_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.current_label.setStyleSheet("background-color: #2E2E2E; padding: 10px; color: white;")
        self.current_label.setWordWrap(True)
        self.current_layout.addWidget(self.current_label)
        self.current_layout.addStretch()
        self.current_scroll_area.setWidget(self.current_widget)
        self.current_scroll_area.setMinimumHeight(200)
        self.deps_layout.addWidget(self.current_scroll_area)

        self.latest_scroll_area = QScrollArea()
        self.latest_scroll_area.setWidgetResizable(True)
        self.latest_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.latest_widget = QWidget()
        self.latest_layout = QVBoxLayout(self.latest_widget)
        self.latest_label = QLabel()
        self.latest_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.latest_label.setStyleSheet("background-color: #2E2E2E; padding: 10px; color: white;")
        self.latest_label.setWordWrap(True)
        self.latest_layout.addWidget(self.latest_label)
        self.latest_layout.addStretch()
        self.latest_scroll_area.setWidget(self.latest_widget)
        self.latest_scroll_area.setMinimumHeight(200)
        self.deps_layout.addWidget(self.latest_scroll_area)

        self.changes_scroll = QScrollArea()
        self.changes_scroll.setWidgetResizable(True)
        self.changes_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.changes_widget = QWidget()
        self.changes_layout = QVBoxLayout(self.changes_widget)

        self.changes_title = QLabel("<b>Changes:</b>")
        self.changes_title.setStyleSheet("font-weight: bold;")
        self.changes_layout.addWidget(self.changes_title)

        self.added_label = QLabel()
        self.added_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.added_label.setWordWrap(True)
        self.added_label.setStyleSheet("padding: 5px;")
        self.removed_label = QLabel()
        self.removed_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.removed_label.setWordWrap(True)
        self.removed_label.setStyleSheet("padding: 5px;")

        self.changes_layout.addWidget(self.added_label)
        self.changes_layout.addWidget(self.removed_label)

        self.changes_layout.addStretch()
        self.changes_scroll.setWidget(self.changes_widget)
        self.main_layout.addWidget(self.changes_scroll)

        self.update_display()

    def update_display(self):
        self.hide_extras = self.hide_extras_checkbox.isChecked()
        filtered_current_deps = self.filter_extras(self.current_deps_full)
        filtered_latest_deps = self.filter_extras(self.latest_deps_full)

        if filtered_current_deps:
            current_text = "\n".join(filtered_current_deps)
        else:
            current_text = "No dependencies found."
        self.current_label.setText(f"<b>Current:</b>\n{current_text}")

        if filtered_latest_deps:
            latest_text = "\n".join(filtered_latest_deps)
        else:
            latest_text = "No dependencies found."
        self.latest_label.setText(f"<b>Latest:</b>\n{latest_text}")

        added = set(filtered_latest_deps) - set(filtered_current_deps)
        removed = set(filtered_current_deps) - set(filtered_latest_deps)

        if added:
            added_text = "<b>Added:</b> " + ", ".join(sorted(added))
            self.added_label.setText(added_text)
            self.added_label.setVisible(True)
        else:
            self.added_label.setText("")
            self.added_label.setVisible(False)

        if removed:
            removed_text = "<b>Removed:</b> " + ", ".join(sorted(removed))
            self.removed_label.setText(removed_text)
            self.removed_label.setVisible(True)
        else:
            self.removed_label.setText("")
            self.removed_label.setVisible(False)

        if not added and not removed:
            self.changes_title.setText("<b>Changes:</b>")
            self.added_label.setText("No changes in dependencies.")
            self.removed_label.setText("")
            self.added_label.setVisible(True)
        else:
            self.changes_title.setText("<b>Changes:</b>")

    def filter_extras(self, deps):
        if not self.hide_extras:
            return deps
        filtered = [dep for dep in deps if 'extra' not in dep.lower()]
        return filtered

class PackageChecker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python Package Checker")
        self.setMinimumSize(800, 1200)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        button_layout = QHBoxLayout()
        self.check_all_button = QPushButton("Check All")
        button_layout.addWidget(self.check_all_button)
        self.check_outdated_button = QPushButton("Check Outdated")
        button_layout.addWidget(self.check_outdated_button)
        self.pip_check_button = QPushButton("Pip Check")
        button_layout.addWidget(self.pip_check_button)
        layout.addLayout(button_layout)

        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Type pip command here to run (e.g., pip install requests)")
        self.command_input.returnPressed.connect(self.execute_command)
        layout.addWidget(self.command_input)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Package", "Current Version", "Latest Version"])
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self.open_context_menu)
        layout.addWidget(self.results_table)

        self.check_all_button.clicked.connect(self.check_all_packages)
        self.check_outdated_button.clicked.connect(self.check_outdated_packages)
        self.pip_check_button.clicked.connect(self.run_pip_check)

        self.current_mode = None
        self.outdated_packages = []
        self.requires_map = {}
        self.required_by_map = {}
        self.current_thread = None

    def verify_installation(self, package_name, expected_version):
        try:
            installed_version = importlib.metadata.version(package_name)
            return installed_version == expected_version
        except importlib.metadata.PackageNotFoundError:
            return False

    def _is_package_available(self, pkg_name: str):
        package_exists = importlib.util.find_spec(pkg_name) is not None
        package_version = "N/A"
        if package_exists:
            try:
                package_version = importlib.metadata.version(pkg_name)
            except importlib.metadata.PackageNotFoundError:
                if pkg_name == "torch":
                    try:
                        package = importlib.import_module(pkg_name)
                        temp_version = getattr(package, "__version__", "N/A")
                        if "dev" in temp_version:
                            package_version = temp_version
                            package_exists = True
                        else:
                            package_exists = False
                    except ImportError:
                        package_exists = False
                else:
                    package_exists = False
        return package_exists, package_version

    def run_pip_check(self):
        self.progress_bar.setVisible(True)
        self.pip_check_button.setEnabled(False)
        self.thread = QThread()
        self.worker = PipCheckWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_pip_check_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_pip_check_finished(self):
        self.progress_bar.setVisible(False)
        self.pip_check_button.setEnabled(True)

    def execute_command(self):
        command = self.command_input.text().strip()
        if not command:
            return

        self.command_input.setEnabled(False)
        self.progress_bar.setVisible(True)

        self.command_input.clear()

        print(f"\n=== Executing Command: {command} ===")

        self.command_thread = QThread()
        self.command_worker = CommandWorker(command)
        self.command_worker.moveToThread(self.command_thread)

        self.command_thread.started.connect(self.command_worker.run)
        self.command_worker.finished.connect(self.on_command_finished)
        self.command_worker.error.connect(self.on_command_error)
        self.command_worker.finished.connect(self.command_thread.quit)
        self.command_worker.error.connect(self.command_thread.quit)
        self.command_worker.finished.connect(self.command_worker.deleteLater)
        self.command_worker.error.connect(self.command_worker.deleteLater)
        self.command_thread.finished.connect(self.command_thread.deleteLater)

        self.command_thread.start()

    def on_command_finished(self, output):
        self.progress_bar.setVisible(False)
        self.command_input.setEnabled(True)
        self.command_input.setFocus()

        print(output)
        print("=== Command Completed Successfully ===\n")

        command_text = self.command_input.placeholderText()
        if self.current_mode == 'all':
            self.check_all_packages()
        elif self.current_mode == 'outdated':
            self.check_outdated_packages()

    def on_command_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.command_input.setEnabled(True)
        self.command_input.setFocus()

        print(f"Error: {error_message}")
        print("=== Command Failed ===\n")

        self.show_message("Command Error", f"Command execution failed:\n{error_message}")

    def check_all_packages(self):
        self.current_mode = 'all'
        self.results_table.clearContents()
        self.results_table.setRowCount(0)
        try:
            installed_packages = list(importlib.metadata.distributions())
            installed_packages.sort(key=lambda x: x.metadata['Name'].lower())
            self.requires_map = {}
            self.required_by_map = {}
            package_names = {}
            for dist in installed_packages:
                name = dist.metadata['Name']
                package_names[name.lower()] = name
                requires = dist.requires or []
                self.requires_map[name] = requires
                for req in requires:
                    req_name = req.split()[0]
                    if req_name in self.required_by_map:
                        self.required_by_map[req_name].append(name)
                    else:
                        self.required_by_map[req_name] = [name]
            self.results_table.setRowCount(len(installed_packages))
            for row, dist in enumerate(installed_packages):
                name = dist.metadata['Name']
                version_installed = dist.version
                self.results_table.setItem(row, 0, QTableWidgetItem(name))
                self.results_table.setItem(row, 1, QTableWidgetItem(version_installed))
                self.results_table.setItem(row, 2, QTableWidgetItem("N/A"))
                self.set_tooltip_for_package(row, name)
            self.show_message("Check All Complete", f"Total packages installed: {len(installed_packages)}")
        except Exception as e:
            self.show_message("Error", f"Error while checking packages: {str(e)}")
        self.results_table.scrollToTop()

    def check_outdated_packages(self):
        self.current_mode = 'outdated'
        self.results_table.clearContents()
        self.results_table.setRowCount(0)
        self.results_table.setSortingEnabled(False)
        self.progress_bar.setVisible(True)
        self.check_outdated_button.setEnabled(False)
        self.thread = QThread()
        self.worker = OutdatedPackagesWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_outdated_packages_checked)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_outdated_packages_checked(self, outdated_packages):
        self.progress_bar.setVisible(False)
        self.check_outdated_button.setEnabled(True)
        if not outdated_packages:
            self.show_message("Up to Date", "All packages are up to date!")
            return
        self.outdated_packages = outdated_packages
        self.requires_map = {}
        self.required_by_map = {}
        for pkg in outdated_packages:
            name = pkg[0]
            try:
                dist = importlib.metadata.distribution(name)
                requires = dist.requires or []
                self.requires_map[name] = requires
                for req in requires:
                    req_name = req.split()[0]
                    if req_name in self.required_by_map:
                        self.required_by_map[req_name].append(name)
                    else:
                        self.required_by_map[req_name] = [name]
            except importlib.metadata.PackageNotFoundError:
                pass
        self.results_table.setRowCount(len(outdated_packages))
        for row, (name, current, latest) in enumerate(outdated_packages):
            self.results_table.setItem(row, 0, QTableWidgetItem(name))
            self.results_table.setItem(row, 1, QTableWidgetItem(current))
            self.results_table.setItem(row, 2, QTableWidgetItem(latest))
            self.set_tooltip_for_package(row, name)
        self.show_message("Outdated Packages", f"Total outdated packages: {len(outdated_packages)}")
        self.results_table.setSortingEnabled(True)
        self.results_table.sortItems(0, Qt.AscendingOrder)
        self.results_table.scrollToTop()

    def compare_dependencies(self, package_name):
        try:
            current_dist = importlib.metadata.distribution(package_name)
            current_version = current_dist.version
            current_requires = current_dist.requires or []
            current_deps = sorted([req for req in current_requires])

            url = f"https://pypi.org/pypi/{package_name}/json"
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status != 200:
                    raise Exception(f"PyPI returned status code {response.status}")
                data = json.load(response)
                latest_version = data['info']['version']
                latest_requires = data['info'].get('requires_dist', []) or []
                latest_deps = sorted([req for req in latest_requires if req])

            dialog = CompareDependenciesDialog(
                self,
                package_name,
                current_version,
                latest_version,
                current_deps,
                latest_deps
            )
            dialog.exec()

        except Exception as e:
            self.show_message("Error", f"Error comparing dependencies: {str(e)}")

    def on_worker_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.check_outdated_button.setEnabled(True)
        self.show_message("Error", f"Error while checking outdated packages: {error_message}")

    def open_context_menu(self, position: QPoint):
        selected_row = self.results_table.currentRow()
        if selected_row < 0:
            return
        package_item = self.results_table.item(selected_row, 0)
        if not package_item:
            return
        package_name = package_item.text()
        menu = QMenu(self)
        upgrade_action = menu.addAction("Upgrade/Downgrade")
        upgrade_action.triggered.connect(lambda: self.fetch_versions(package_name, position))
        info_action = menu.addAction("View Package Info")
        info_action.triggered.connect(lambda: self.show_package_info(package_name))
        deps_action = menu.addAction("Show Reverse Dependencies")
        deps_action.triggered.connect(lambda: self.show_reverse_dependencies(package_name))
        compare_deps_action = menu.addAction("Compare Dependencies")
        compare_deps_action.triggered.connect(lambda: self.compare_dependencies(package_name))
        menu.exec(self.results_table.viewport().mapToGlobal(position))

    def show_reverse_dependencies(self, package_name):
        try:
            command = ["pipdeptree", "--reverse", "--packages", package_name, "--depth", "1"]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

            print("\n=== Reverse Dependencies for", package_name, "===")
            print(stdout)
            if stderr:
                print("Errors/Warnings:")
                print(stderr)
            print("=====================================\n")
        except Exception as e:
            print(f"Error running pipdeptree: {str(e)}")

    def fetch_versions(self, package_name, position):
        self.package_name_to_upgrade = package_name
        self.position_for_menu = position

        self.thread_versions = QThread()
        self.worker_versions = VersionsWorker(package_name)
        self.worker_versions.moveToThread(self.thread_versions)
        self.thread_versions.started.connect(self.worker_versions.run)
        self.worker_versions.finished.connect(self.on_versions_fetched)
        self.worker_versions.error.connect(self.on_versions_error)
        self.worker_versions.finished.connect(self.thread_versions.quit)
        self.worker_versions.finished.connect(self.worker_versions.deleteLater)
        self.worker_versions.error.connect(self.thread_versions.quit)
        self.worker_versions.error.connect(self.worker_versions.deleteLater)
        self.thread_versions.finished.connect(self.thread_versions.deleteLater)
        self.thread_versions.start()

    def on_versions_fetched(self, versions):
        self.show_versions_menu(self.package_name_to_upgrade, versions, self.position_for_menu)

    def show_versions_menu(self, package_name, versions, position):
        if not versions:
            self.show_message("No Versions Found", f"No available versions found for '{package_name}'.")
            return

        menu = QMenu(self)
        for ver, release_date in reversed(versions):
            action_text = f"{ver} ({release_date})"
            action = menu.addAction(action_text)
            action.triggered.connect(lambda checked, v=ver: self.upgrade_downgrade_package(package_name, v))

        menu.exec(self.results_table.viewport().mapToGlobal(position))

    def upgrade_downgrade_package(self, package_name, selected_version):
        reply = QMessageBox.question(
            self,
            "Confirm Upgrade/Downgrade",
            f"Are you sure you want to install version {selected_version} of '{package_name}'?\n\nThis will not install dependencies.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        self.progress_bar.setVisible(True)
        self.results_table.setEnabled(False)
        self.package_name_being_updated = package_name
        self.selected_version = selected_version

        self.thread_pip = QThread()
        self.worker_pip = PipWorker(package_name, selected_version)
        self.worker_pip.moveToThread(self.thread_pip)

        self.thread_pip.started.connect(self.worker_pip.run)
        self.worker_pip.finished.connect(self.on_pip_finished)
        self.worker_pip.error.connect(self.on_pip_error)
        self.worker_pip.finished.connect(self.thread_pip.quit)
        self.worker_pip.error.connect(self.thread_pip.quit)
        self.worker_pip.finished.connect(self.worker_pip.deleteLater)
        self.worker_pip.error.connect(self.worker_pip.deleteLater)
        self.thread_pip.finished.connect(self.thread_pip.deleteLater)

        self.thread_pip.start()

    def on_pip_finished(self, output):
        self.progress_bar.setVisible(False)
        self.results_table.setEnabled(True)

        if not self.verify_installation(self.package_name_being_updated, self.selected_version):
            self.show_message("Error", f"Package '{self.package_name_being_updated}' installation verification failed.")
            return

        self.show_message("Success", f"Package '{self.package_name_being_updated}' upgraded/downgraded successfully.\n\nOutput:\n{output}")

        if self.current_mode == 'outdated':
            self.thread_latest = QThread()
            self.worker_latest = LatestVersionWorker(self.package_name_being_updated)
            self.worker_latest.moveToThread(self.thread_latest)
            self.thread_latest.started.connect(self.worker_latest.run)
            self.worker_latest.finished.connect(self.on_latest_version_fetched)
            self.worker_latest.error.connect(self.on_latest_version_error)
            self.worker_latest.finished.connect(self.thread_latest.quit)
            self.worker_latest.finished.connect(self.worker_latest.deleteLater)
            self.worker_latest.error.connect(self.thread_latest.quit)
            self.worker_latest.error.connect(self.worker_latest.deleteLater)
            self.thread_latest.finished.connect(self.thread_latest.deleteLater)
            self.thread_latest.start()
        else:
            self.check_all_packages()

    def on_pip_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.results_table.setEnabled(True)
        self.show_message("Error", f"Error while upgrading/downgrading '{self.package_name_being_updated}':\n{error_message}")

    def on_latest_version_fetched(self, latest_version):
        self.update_outdated_after_upgrade(self.package_name_being_updated, latest_version)

    def on_latest_version_error(self, error_message):
        self.show_message("Error", f"Error fetching latest version for '{self.package_name_being_updated}': {error_message}")

    def update_outdated_after_upgrade(self, package_name, latest_version):
        try:
            installed_version = importlib.metadata.version(package_name)
            if version.parse(installed_version) >= version.parse(latest_version):
                self.outdated_packages = [pkg for pkg in self.outdated_packages if pkg[0].lower() != package_name.lower()]
                self.remove_package_from_table(package_name)
            else:
                for i, pkg in enumerate(self.outdated_packages):
                    if pkg[0].lower() == package_name.lower():
                        self.outdated_packages[i] = (pkg[0], installed_version, latest_version)
                        row = self.find_row(package_name)
                        if row is not None:
                            self.results_table.setItem(row, 1, QTableWidgetItem(installed_version))
                            self.results_table.setItem(row, 2, QTableWidgetItem(latest_version))
                            self.set_tooltip_for_package(row, package_name)
            self.show_message("Update Complete", f"Package '{package_name}' has been updated in the outdated list.")
        except importlib.metadata.PackageNotFoundError:
            self.show_message("Error", f"Package '{package_name}' not found after installation.")

    def find_row(self, package_name):
        for row in range(self.results_table.rowCount()):
            item = self.results_table.item(row, 0)
            if item and item.text().lower() == package_name.lower():
                return row
        return None

    def remove_package_from_table(self, package_name):
        row = self.find_row(package_name)
        if row is not None:
            self.results_table.removeRow(row)

    def on_versions_error(self, error_message):
        self.show_message("Error", f"Error while fetching versions: {error_message}")

    def show_package_info(self, package_name):
        url = f"https://pypi.org/pypi/{package_name}/json"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status != 200:
                    raise Exception(f"PyPI returned status code {response.status}")
                data = json.load(response)
                info = data['info']
                description = info.get('summary', 'No description available.')
                author = info.get('author', 'N/A')
                homepage = info.get('home_page', 'N/A')
                package_url = info.get('package_url', f"https://pypi.org/project/{package_name}/")
                project_urls = info.get('project_urls', {})
                documentation = project_urls.get('Documentation', 'N/A')
                info_dialog = QDialog(self)
                info_dialog.setWindowTitle(f"Package Info: {package_name}")
                layout = QVBoxLayout(info_dialog)
                layout.addWidget(QLabel(f"<b>Package:</b> {package_name}"))
                layout.addWidget(QLabel(f"<b>Author:</b> {author}"))
                homepage_label = QLabel(f"<b>Homepage:</b> <a href='{homepage}'>{homepage}</a>")
                homepage_label.setTextFormat(Qt.RichText)
                homepage_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
                homepage_label.setOpenExternalLinks(True)
                layout.addWidget(homepage_label)
                pypi_label = QLabel(f"<b>PyPI Page:</b> <a href='{package_url}'>{package_url}</a>")
                pypi_label.setTextFormat(Qt.RichText)
                pypi_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
                pypi_label.setOpenExternalLinks(True)
                layout.addWidget(pypi_label)
                if documentation != 'N/A':
                    doc_label = QLabel(f"<b>Documentation:</b> <a href='{documentation}'>{documentation}</a>")
                    doc_label.setTextFormat(Qt.RichText)
                    doc_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
                    doc_label.setOpenExternalLinks(True)
                    layout.addWidget(doc_label)
                description_label = QLabel(f"<b>Description:</b> {description}")
                description_label.setWordWrap(True)
                layout.addWidget(description_label)
                info_dialog.setLayout(layout)
                info_dialog.exec()
        except urllib.error.URLError as e:
            self.show_message("Error", f"Network error: {str(e)}")
        except TimeoutError:
            self.show_message("Error", "Connection timed out")
        except json.JSONDecodeError:
            self.show_message("Error", "Invalid response from PyPI")
        except Exception as e:
            self.show_message("Error", f"Error fetching package info: {str(e)}")

    def show_message(self, title, message):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        icon = QMessageBox.Information if title != "Error" else QMessageBox.Critical
        msg_box.setIcon(icon)
        msg_box.exec()

    def set_tooltip_for_package(self, row, package_name):
        requires = self.requires_map.get(package_name, [])
        required_by = self.required_by_map.get(package_name, [])
        requires_text = ", ".join([req.split()[0] for req in requires]) if requires else "None"
        required_by_text = ", ".join(required_by) if required_by else "None"
        tooltip_text = f"Requires: {requires_text}\nRequired by: {required_by_text}"
        package_item = self.results_table.item(row, 0)
        package_item.setToolTip(tooltip_text)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = PackageChecker()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()