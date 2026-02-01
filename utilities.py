# utilities.py - GUI-dependent utilities
from pathlib import Path
import os
import platform
import subprocess

from PySide6.QtCore import QRunnable, QObject, Signal
from PySide6.QtWidgets import QMessageBox

from utilities_core import (
    set_cuda_paths,
    clean_triton_cache,
    supports_flash_attention,
    get_model_native_precision,
    get_appropriate_dtype,
    get_embedding_batch_size,
    my_cprint,
    get_embedding_dtype_and_batch,
    configure_logging,
)

from config import get_config

def format_citations(metadata_list):
    def group_metadata(metadata_list):
        grouped = {}
        for metadata in metadata_list:
            file_path = metadata['file_path']
            grouped.setdefault(file_path, {
                'name': Path(file_path).name,
                'scores': [],
                'pages': set(),
                'file_type': metadata.get('file_type', '')
            })
            grouped[file_path]['scores'].append(metadata['similarity_score'])
            if grouped[file_path]['file_type'] == '.pdf':
                page_number = metadata.get('page_number')
                if page_number is not None:
                    grouped[file_path]['pages'].add(page_number)
        return grouped

    def format_pages(pages):
        if not pages:
            return ''
        sorted_pages = sorted(pages)
        ranges = []
        start = prev = sorted_pages[0]
        for page in sorted_pages[1:]:
            if page == prev + 1:
                prev = page
            else:
                ranges.append((start, prev))
                start = prev = page
        ranges.append((start, prev))
        page_str = ', '.join(f"{s}-{e}" if s != e else f"{s}" for s, e in ranges)
        return f'<span style="color:#666;"> p.{page_str}</span>'

    def create_citation(data, file_path):
        min_score = min(data['scores'])
        max_score = max(data['scores'])
        score_range = f"{min_score:.4f}" if min_score == max_score else f"{min_score:.4f}-{max_score:.4f}"
        pages_html = format_pages(data['pages']) if data['file_type'] == '.pdf' else ''
        citation = (
            f'<a href="file:{file_path}" style="color:#DAA520;text-decoration:none;">{data["name"]}</a>'
            f'<span style="color:#808080;font-size:0.9em;"> ['
            f'<span style="color:#4CAF50;">{score_range}</span>]'
            f'{pages_html}'
            f'</span>'
        )
        return min_score, citation

    grouped_citations = group_metadata(metadata_list)
    citations_with_scores = [create_citation(data, file_path) for file_path, data in grouped_citations.items()]
    sorted_citations = [citation for _, citation in sorted(citations_with_scores)]
    list_items = "".join(f"<li>{citation}</li>" for citation in sorted_citations)

    return f"<ol>{list_items}</ol>"

def backup_database_incremental(new_database_name):
    import shutil
    import logging
    logger = logging.getLogger(__name__)

    config = get_config()
    source_directory = config.vector_db_dir
    backup_directory = config.vector_db_backup_dir

    backup_directory.mkdir(parents=True, exist_ok=True)

    source_db_path = source_directory / new_database_name
    backup_db_path = backup_directory / new_database_name

    if backup_db_path.exists():
        logger.info(f"Existing backup found for {new_database_name} - attempting to remove")
        try:
            shutil.rmtree(backup_db_path)
            logger.info("Successfully removed existing backup")
        except Exception as e:
            logger.warning(f"Failed to remove existing backup: {e}")

    try:
        shutil.copytree(source_db_path, backup_db_path)
        logger.info(f"Successfully created backup of {new_database_name}")
    except Exception as e:
        logger.error(f"Backup failed: {e}")

def open_file(file_path):
    try:
        if platform.system() == "Windows":
            os.startfile(file_path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", file_path])
        else:
            subprocess.Popen(["xdg-open", file_path])
    except OSError:
        QMessageBox.warning(None, "Error", "No default viewer detected.")

def open_settings_dialog():
    from PySide6.QtWidgets import QApplication
    for widget in QApplication.topLevelWidgets():
        if hasattr(widget, "show_settings_dialog") and callable(widget.show_settings_dialog):
            widget.show_settings_dialog()
            break

def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError:
        QMessageBox.warning(None, "Unable to delete file(s), please delete manually.")

def check_preconditions_for_db_creation(database_name, skip_ocr=False):
    import torch
    config = get_config()

    if not database_name or len(database_name) < 3 or database_name.lower() in ["null", "none"]:
        return False, "Name must be at least 3 characters long and not be 'null' or 'none.'"

    vector_db_path = config.vector_db_dir / database_name
    if vector_db_path.exists():
        return False, (
            f"A vector database called '{database_name}' already exists—"
            "choose a different name or delete the old one first."
        )

    embedding_model_name = config.EMBEDDING_MODEL_NAME
    if not embedding_model_name:
        return False, "You must first download an embedding model, select it, and choose documents before proceeding."

    if not any(file.is_file() for file in config.docs_dir.iterdir()):
        return False, "No documents are yet added to be processed."

    compute_device = config.Compute_Device.available
    database_creation = config.Compute_Device.database_creation
    if ("cuda" in compute_device or "mps" in compute_device) and database_creation == "cpu":
        return False, ("GPU-acceleration is available and strongly recommended. "
                       "Please switch the database creation device to 'cuda' or 'mps', "
                       "or confirm your choice in the GUI.")

    if not torch.cuda.is_available() and config.database.half:
        return False, ("CUDA is not available on your system, but half-precision (FP16) "
                       "is selected for database creation. Half-precision requires CUDA. "
                       "Please disable half-precision in the configuration or use a CUDA-enabled GPU.")

    return True, ""