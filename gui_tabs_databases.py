# gui_tabs_databases.py

import time
import gc
import json
from pathlib import Path
import yaml
from PySide6.QtCore import QDir, QRegularExpression, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QRegularExpressionValidator
from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QTreeView, QFileSystemModel, QMenu, QGroupBox, QLineEdit, QGridLayout, QSizePolicy, QComboBox, QToolButton

from choose_documents_and_vector_model import choose_documents_directory
from utilities import check_preconditions_for_db_creation, open_file, delete_file, backup_database_incremental, my_cprint, open_settings_dialog
from download_model import model_downloaded_signal
from constants import TOOLTIPS
from config import get_config

class VectorDBWorker(QThread):
    finished = Signal(bool, str)
    progress = Signal(str)

    def __init__(self, database_name, parent=None):
        super().__init__(parent)
        self.database_name = database_name
        self._is_cancelled = False

    def run(self):
        try:
            from vector_db_creator import CreateVectorDB

            self.progress.emit("Initializing database creation...")

            create_vector_db = CreateVectorDB(database_name=self.database_name)
            create_vector_db.run()

            if not self._is_cancelled:
                self.finished.emit(True, "Database created successfully!")

        except FileExistsError as e:
            self.finished.emit(False, str(e))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, f"Database creation failed: {str(e)}")
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
            gc.collect()

    def cancel(self):
        self._is_cancelled = True


class CustomFileSystemModel(QFileSystemModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFilter(QDir.Files)


class DatabasesTab(QWidget):
    def __init__(self):
        super().__init__()
        model_downloaded_signal.downloaded.connect(self.update_model_combobox)
        self.layout = QVBoxLayout(self)
        self.documents_group_box = self.create_group_box("Files To Add to Database", "Docs_for_DB")
        self.groups = {self.documents_group_box: 1}
        grid_layout_top_buttons = QGridLayout()

        self.choose_docs_button = QPushButton("Choose Files")
        self.choose_docs_button.setToolTip(TOOLTIPS["CHOOSE_FILES"])
        self.choose_docs_button.clicked.connect(choose_documents_directory)

        self.model_combobox = QComboBox()
        self.model_combobox.setToolTip(TOOLTIPS["SELECT_VECTOR_MODEL"])
        self.populate_model_combobox()
        self.model_combobox.currentIndexChanged.connect(self.on_model_selected)
        self.model_combobox.activated.connect(self.refresh_model_combobox)

        self.create_settings_btn = QToolButton()
        self.create_settings_btn.setText("⚙")
        self.create_settings_btn.setToolTip("Open Database Creation settings")
        self.create_settings_btn.clicked.connect(open_settings_dialog)

        self.create_db_button = QPushButton("Create Vector Database")
        self.create_db_button.setToolTip(TOOLTIPS["CREATE_VECTOR_DB"])
        self.create_db_button.clicked.connect(self.on_create_db_clicked)
        self.create_db_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        grid_layout_top_buttons.addWidget(self.choose_docs_button, 0, 0)
        grid_layout_top_buttons.addWidget(self.model_combobox, 0, 1)
        grid_layout_top_buttons.addWidget(self.create_settings_btn, 0, 2)
        grid_layout_top_buttons.addWidget(self.create_db_button, 0, 3)

        number_of_columns = 4
        for column_index in range(number_of_columns):
            grid_layout_top_buttons.setColumnStretch(column_index, 1)

        hbox2 = QHBoxLayout()
        self.database_name_input = QLineEdit()
        self.database_name_input.setToolTip(TOOLTIPS["DATABASE_NAME_INPUT"])
        self.database_name_input.setPlaceholderText("Enter database name")
        self.database_name_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        regex = QRegularExpression("^[a-z0-9_-]*$")
        validator = QRegularExpressionValidator(regex, self.database_name_input)
        self.database_name_input.setValidator(validator)
        hbox2.addWidget(self.database_name_input)
        self.layout.addLayout(grid_layout_top_buttons)
        self.layout.addLayout(hbox2)
        self.sync_combobox_with_config()

        self.db_worker = None
        self.current_model_name = None
        self.current_database_name = None
        self.process_start_time = None

    def _validation_failed(self, message: str):
        QMessageBox.warning(self, "Validation Failed", message)
        self.reenable_create_db_button()

    def refresh_model_combobox(self, index):
        current_text = self.model_combobox.currentText()
        self.populate_model_combobox()
        idx = self.model_combobox.findText(current_text)
        if idx >= 0:
            self.model_combobox.setCurrentIndex(idx)

    def update_model_combobox(self, model_name, model_type):
        if model_type == "vector":
            self.populate_model_combobox()
            self.sync_combobox_with_config()

    def populate_model_combobox(self):
        self.model_combobox.clear()
        self.model_combobox.addItem("Select a model", None)
        config = get_config()
        vector_dir = config.vector_models_dir
        if not vector_dir.exists():
            return
        for folder in vector_dir.iterdir():
            if folder.is_dir():
                display_name = folder.name
                full_path = str(folder)
                self.model_combobox.addItem(display_name, full_path)

    def sync_combobox_with_config(self):
        config = get_config()
        current_model = config.EMBEDDING_MODEL_NAME
        if current_model:
            model_index = self.model_combobox.findData(current_model)
            if model_index != -1:
                self.model_combobox.setCurrentIndex(model_index)
            else:
                self.model_combobox.setCurrentIndex(0)
        else:
            self.model_combobox.setCurrentIndex(0)

    def on_model_selected(self, index):
        selected_path = self.model_combobox.itemData(index)
        config = get_config()

        embedding_dimensions = None
        if selected_path:
            config.EMBEDDING_MODEL_NAME = selected_path

            sp_lower = str(selected_path).lower()
            if "stella" in sp_lower or "static-retrieval" in sp_lower:
                embedding_dimensions = 1024
            else:
                config_json_path = Path(selected_path) / "config.json"
                if config_json_path.exists():
                    with open(config_json_path, 'r', encoding='utf-8') as json_file:
                        model_config = json.load(json_file)
                    val = model_config.get("hidden_size") or model_config.get("d_model")
                    if isinstance(val, int):
                        embedding_dimensions = val

            if embedding_dimensions:
                config.EMBEDDING_MODEL_DIMENSIONS = embedding_dimensions
        else:
            config.EMBEDDING_MODEL_NAME = None
            config.EMBEDDING_MODEL_DIMENSIONS = None

        config.save()

    def create_group_box(self, title, directory_name):
        group_box = QGroupBox(title)
        layout = QVBoxLayout()
        tree_view = self.setup_directory_view(directory_name)
        layout.addWidget(tree_view)
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        group_box.toggled.connect(lambda checked, gb=group_box: self.toggle_group_box(gb, checked))
        return group_box

    def _refresh_docs_model(self):
        if hasattr(self, 'docs_model') and self.docs_model:
            docs_path = str(Path(__file__).parent / "Docs_for_DB")
            root_index = self.docs_model.index(docs_path)
            if self.docs_model.canFetchMore(root_index):
                self.docs_model.fetchMore(root_index)

    def setup_directory_view(self, directory_name):
        tree_view = QTreeView()
        model = CustomFileSystemModel()
        tree_view.setModel(model)
        tree_view.setSelectionMode(QTreeView.ExtendedSelection)

        script_dir = Path(__file__).resolve().parent
        config = get_config()
        if directory_name == "Docs_for_DB":
            directory_path = config.docs_dir

        directory_path.mkdir(parents=True, exist_ok=True)

        model.setRootPath(str(directory_path))
        tree_view.setRootIndex(model.index(str(directory_path)))

        tree_view.hideColumn(1)
        tree_view.hideColumn(2)
        tree_view.hideColumn(3)
        tree_view.doubleClicked.connect(self.on_double_click)
        tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        tree_view.customContextMenuRequested.connect(self.on_context_menu)

        if directory_name == "Docs_for_DB":
            self.docs_model = model
            self.docs_view = tree_view
            self.docs_refresh = QTimer(self)
            self.docs_refresh.setInterval(500)
            self.docs_refresh.timeout.connect(self._refresh_docs_model)
            self.docs_refresh.start()
        return tree_view

    def on_double_click(self, index):
        tree_view = self.sender()
        model = tree_view.model()
        file_path = model.filePath(index)
        open_file(file_path)

    def on_context_menu(self, point):
        tree_view = self.sender()
        context_menu = QMenu(self)
        delete_action = QAction("Delete File", self)
        context_menu.addAction(delete_action)
        delete_action.triggered.connect(lambda: self.on_delete_file(tree_view))
        context_menu.exec_(tree_view.viewport().mapToGlobal(point))

    def on_delete_file(self, tree_view):
        selected_indexes = tree_view.selectedIndexes()
        model = tree_view.model()
        for index in selected_indexes:
            if index.column() == 0:
                file_path = model.filePath(index)
                delete_file(file_path)

    def on_create_db_clicked(self):
        if self.model_combobox.currentIndex() == 0:
            QMessageBox.warning(self, "No Model Selected", "Please select a model before creating a database.")
            return

        self.create_db_button.setDisabled(True)
        self.choose_docs_button.setDisabled(True)
        self.model_combobox.setDisabled(True)
        self.database_name_input.setDisabled(True)

        database_name = self.database_name_input.text().strip()
        model_name = self.model_combobox.currentText()

        self.current_database_name = database_name
        self.current_model_name = model_name

        self.start_database_creation(database_name, model_name)

    def start_database_creation(self, database_name, model_name):
        try:
            ok, msg = check_preconditions_for_db_creation(database_name)
            if not ok:
                self._validation_failed(msg)
                return

            self.db_worker = VectorDBWorker(database_name, parent=self)
            self.db_worker.progress.connect(self.on_worker_progress)
            self.db_worker.finished.connect(self.on_worker_finished)
            self.db_worker.start()

            self.process_start_time = time.time()
            my_cprint(f"Started database creation for: {database_name}", "green")

        except Exception as e:
            self._validation_failed(f"Failed to start database creation: {str(e)}")

    def on_worker_progress(self, message):
        my_cprint(message, "cyan")

    def on_worker_finished(self, success, message):
        try:
            if success:
                my_cprint(f"{self.current_model_name} removed from memory.", "red")

                config = get_config()
                config.add_database(
                    self.current_database_name,
                    config.EMBEDDING_MODEL_NAME,
                    config.database.chunk_size,
                    config.database.chunk_overlap
                )

                backup_database_incremental(self.current_database_name)

                if hasattr(self, 'docs_model') and self.docs_model:
                    docs_path = str(config.docs_dir)
                    self.docs_model.setRootPath("")
                    self.docs_model.setRootPath(docs_path)
                    if hasattr(self, 'docs_view') and self.docs_view:
                        self.docs_view.setRootIndex(self.docs_model.index(docs_path))

                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.critical(self, "Error", message)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in completion handler: {str(e)}")

        finally:
            self._cleanup_worker()
            self.reenable_create_db_button()

    def _cleanup_worker(self):
        if self.db_worker:
            if self.db_worker.isRunning():
                self.db_worker.cancel()
                self.db_worker.wait(5000)
                if self.db_worker.isRunning():
                    self.db_worker.terminate()
                    self.db_worker.wait()
            self.db_worker = None

    def reenable_create_db_button(self):
        self.create_db_button.setDisabled(False)
        self.choose_docs_button.setDisabled(False)
        self.model_combobox.setDisabled(False)
        self.database_name_input.setDisabled(False)

        self.current_database_name = None
        self.current_model_name = None
        self.process_start_time = None

        gc.collect()

    def closeEvent(self, event):
        self._cleanup_worker()
        if hasattr(self, 'docs_refresh'):
            self.docs_refresh.stop()
        event.accept()

    def cleanup(self):
        self._cleanup_worker()
        if hasattr(self, 'docs_refresh'):
            self.docs_refresh.stop()

    def toggle_group_box(self, group_box, checked):
        self.groups[group_box] = 1 if checked else 0
        self.adjust_stretch()

    def adjust_stretch(self):
        for group, stretch in self.groups.items():
            self.layout.setStretchFactor(group, stretch if group.isChecked() else 0)