from pathlib import Path
import threading

from PySide6.QtCore import QElapsedTimer, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFileSystemModel,
    QHBoxLayout,
    QProgressDialog,
    QVBoxLayout,
    QDialog,
    QTextEdit,
    QPushButton,
    QMessageBox,
)

from create_symlinks import _create_single_symlink
from config import get_config
from process_manager import get_process_manager

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".epub", ".txt", ".enex", ".eml", ".msg",
    ".csv", ".xls", ".xlsx", ".rtf", ".odt", ".html", ".htm", ".md",
}

CONFIG_FILE = "config.yaml"


class SymlinkWorker(QThread):
    progress = Signal(int)
    finished = Signal(int, list)
    request_model_pause = Signal()
    request_model_resume = Signal()

    def __init__(self, source, target_dir, parent=None):
        super().__init__(parent)
        self.source = source
        self.target_dir = Path(target_dir)
        self._should_stop = False
        self._stop_lock = threading.Lock()

    def run(self):
        self.request_model_pause.emit()

        try:
            if isinstance(self.source, (str, Path)):
                dir_path = Path(self.source)
                try:
                    import os
                    filenames = os.listdir(str(dir_path))
                    files = [
                        str(dir_path / filename)
                        for filename in filenames
                        if (dir_path / filename).is_file() 
                        and (dir_path / filename).suffix.lower() in ALLOWED_EXTENSIONS
                    ]
                except OSError:
                    files = []
                    print(f"Error accessing directory {dir_path}")
            else:
                files = list(self.source)

            total = len(files)
            made = 0
            errors = []
            last_pct = -1
            timer = QElapsedTimer()
            timer.start()
            step = max(1, total // 100) if total else 1

            for i, f in enumerate(files, 1):
                with self._stop_lock:
                    if self._should_stop:
                        break

                ok, err = _create_single_symlink((f, str(self.target_dir)))
                if ok:
                    made += 1
                if err:
                    errors.append(err)
                if i % step == 0 or i == total:
                    pct = int(i * 100 / total) if total else 100
                    if pct != last_pct and timer.elapsed() > 500:
                        self.progress.emit(pct)
                        last_pct = pct
                        timer.restart()

            self.finished.emit(made, errors)
        finally:
            self.request_model_resume.emit()

    def stop(self):
        with self._stop_lock:
            self._should_stop = True


def choose_documents_directory():
    current_dir = Path(__file__).parent.resolve()
    config = get_config()
    target_dir = config.docs_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    msg_box = QMessageBox()
    msg_box.setWindowTitle("Selection Type")
    msg_box.setText("Would you like to select a directory or individual files?")

    dir_button = msg_box.addButton("Select Directory", QMessageBox.ActionRole)
    files_button = msg_box.addButton("Select Files", QMessageBox.ActionRole)
    cancel_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)

    msg_box.exec()
    clicked_button = msg_box.clickedButton()

    if clicked_button == cancel_button:
        return

    file_dialog = QFileDialog()

    def start_worker(source):
        main_window = _get_main_window()

        progress = QProgressDialog("Creating symlinks...", "Cancel", 0, 0, parent=main_window)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        worker = SymlinkWorker(source, target_dir, parent=main_window)

        if main_window is not None:
            main_window._symlink_worker = worker
            main_window._symlink_progress = progress

        def pause_model_watching():
            if main_window and hasattr(main_window, "databases_tab"):
                db_tab = main_window.databases_tab
                if hasattr(db_tab, "docs_refresh"):
                    db_tab.docs_refresh.stop()
                if hasattr(db_tab, "docs_model") and db_tab.docs_model:
                    if hasattr(QFileSystemModel, "DontWatchForChanges"):
                        db_tab.docs_model.setOption(QFileSystemModel.DontWatchForChanges, True)

        def resume_model_watching():
            if main_window and hasattr(main_window, "databases_tab"):
                db_tab = main_window.databases_tab
                if hasattr(db_tab, "docs_model") and db_tab.docs_model:
                    docs_path = str(config.docs_dir)
                    db_tab.docs_model.setRootPath("")
                    db_tab.docs_model.setRootPath(docs_path)

                    if hasattr(QFileSystemModel, "DontWatchForChanges"):
                        db_tab.docs_model.setOption(QFileSystemModel.DontWatchForChanges, False)

                if hasattr(db_tab, "docs_refresh"):
                    db_tab.docs_refresh.start()

        worker.request_model_pause.connect(pause_model_watching)
        worker.request_model_resume.connect(resume_model_watching)

        def on_cancel():
            worker.stop()
            worker.wait(2000)

        progress.canceled.connect(on_cancel)

        def update_progress(pct):
            if progress.maximum() == 0:
                progress.setRange(0, 100)
            progress.setValue(pct)
        worker.progress.connect(update_progress)

        def _done(count, errs):
            progress.reset()
            progress.close()

            msg = f"Created {count} symlinks"
            if errs:
                msg += f" — {len(errs)} errors (see console)"
                print(*errs, sep="\n")
            QMessageBox.information(None, "Symlinks", msg)

            if main_window is not None:
                for attr in ("_symlink_worker", "_symlink_progress"):
                    if hasattr(main_window, attr):
                        try:
                            delattr(main_window, attr)
                        except Exception:
                            pass

        worker.finished.connect(_done)
        worker.start()

    if clicked_button == dir_button:
        file_dialog.setFileMode(QFileDialog.Directory)
        file_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        selected_dir = file_dialog.getExistingDirectory(
            None, "Choose Directory for Database", str(current_dir)
        )
        if selected_dir:
            start_worker(Path(selected_dir))
    else:
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_paths = file_dialog.getOpenFileNames(
            None, "Choose Documents for Database", str(current_dir)
        )[0]
        if file_paths:
            compatible_files = []
            incompatible_files = []
            for file_path in file_paths:
                path = Path(file_path)
                if path.suffix.lower() in ALLOWED_EXTENSIONS:
                    compatible_files.append(str(path))
                else:
                    incompatible_files.append(path.name)

            if incompatible_files and not show_incompatible_files_dialog(incompatible_files):
                return

            if compatible_files:
                start_worker(compatible_files)


def show_incompatible_files_dialog(incompatible_files):
    dialog_text = (
        "The following files are not supported by the database builder and will be skipped:\n\n"
        + "\n".join(incompatible_files)
        + "\n\nSupported types include: PDF, DOC/DOCX, TXT, EPUB, ENEX, EML/MSG, CSV, XLS/XLSX, RTF, ODT, HTML/HTM, and MD."
        "\n\nClick 'OK' to proceed with the supported documents only, or 'Cancel' to go back."
    )

    incompatible_dialog = QDialog()
    incompatible_dialog.resize(800, 600)
    incompatible_dialog.setWindowTitle("Unsupported Files Detected")

    layout = QVBoxLayout()
    text_edit = QTextEdit()
    text_edit.setReadOnly(True)
    text_edit.setText(dialog_text)
    layout.addWidget(text_edit)

    button_box = QHBoxLayout()
    ok_button = QPushButton("OK")
    cancel_button = QPushButton("Cancel")
    button_box.addWidget(ok_button)
    button_box.addWidget(cancel_button)

    layout.addLayout(button_box)
    incompatible_dialog.setLayout(layout)

    ok_button.clicked.connect(incompatible_dialog.accept)
    cancel_button.clicked.connect(incompatible_dialog.reject)

    return incompatible_dialog.exec() == QDialog.Accepted


def load_config():
    return get_config()

def select_embedding_model_directory():
    initial_dir = Path("Models") if Path("Models").exists() else Path.home()
    chosen_directory = QFileDialog.getExistingDirectory(
        None, "Select Embedding Model Directory", str(initial_dir)
    )
    if chosen_directory:
        config = get_config()
        config.EMBEDDING_MODEL_NAME = chosen_directory
        config.save()

def _get_main_window():
    for widget in QApplication.topLevelWidgets():
        if hasattr(widget, "databases_tab"):
            return widget
    return None