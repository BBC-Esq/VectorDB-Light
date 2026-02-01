# document_processor.py

import os
import logging
import warnings
import math
from pathlib import Path, PurePath
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    EverNoteLoader,
    UnstructuredEPubLoader,
    UnstructuredEmailLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredRTFLoader,
    UnstructuredODTLoader,
    UnstructuredMarkdownLoader,
    BSHTMLLoader
)

from typing import Optional, Any, Iterator, Union, List, Tuple
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers import PyMuPDFParser
import fitz
import datetime
import hashlib
import re

from constants import DOCUMENT_LOADERS
from config import get_config

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

INGEST_THREADS = max(2, (os.cpu_count() or 4) - 2)

logger = logging.getLogger(__name__)


def compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def compute_file_hash(file_path):
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def extract_common_metadata(file_path, content_hash=None):
    file_path = os.path.realpath(file_path)
    file_name = os.path.basename(file_path)
    file_type = os.path.splitext(file_path)[1]
    creation_date = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
    modification_date = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()

    file_hash = content_hash if content_hash else compute_file_hash(file_path)

    metadata = {
        "file_path": file_path,
        "file_type": file_type,
        "file_name": file_name,
        "creation_date": creation_date,
        "modification_date": modification_date,
        "hash": file_hash
    }

    return metadata


def extract_document_metadata(file_path, content_hash=None):
    metadata = extract_common_metadata(file_path, content_hash)
    metadata["document_type"] = "document"
    return metadata


class FixedSizeTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, docs: List[Document]) -> List[Document]:
        chunks: List[Document] = []
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = 1

        for doc in docs:
            text = doc.page_content

            if text is None:
                logger.warning(f"Skipping document with None page_content")
                continue

            if isinstance(text, (list, tuple)):
                text = " ".join(str(item) for item in text if item)
                logger.warning(f"Flattened list/tuple page_content to string")

            if not isinstance(text, str):
                text = str(text)

            text = text.strip()

            if not text:
                logger.warning(f"Skipping document with empty page_content")
                continue

            for start in range(0, len(text), step):
                piece = text[start:start + self.chunk_size].strip()

                if not piece:
                    continue

                metadata = doc.metadata if doc.metadata else {}
                chunks.append(Document(page_content=piece, metadata=dict(metadata)))

        return chunks


class CustomPyMuPDFParser(PyMuPDFParser):
    def _lazy_parse(self, blob: Blob, text_kwargs: Optional[dict[str, Any]] = None) -> Iterator[Document]:
        with PyMuPDFParser._lock:
            if blob.path:
                doc = fitz.open(blob.path)
            else:
                with blob.as_bytes_io() as bio:
                    doc = fitz.open(stream=bio, filetype="pdf")

            full_content = []
            for page in doc:
                page_content = self._get_page_content(doc, page, text_kwargs)
                if page_content.strip():
                    full_content.append(f"[[page{page.number + 1}]]{page_content}")

            yield Document(
                page_content="".join(full_content),
                metadata=self._extract_metadata(doc, blob)
            )


class CustomPyMuPDFLoader(PyMuPDFLoader):
    def __init__(self, file_path: Union[str, PurePath], **kwargs: Any) -> None:
        super().__init__(file_path, **kwargs)
        self.parser = CustomPyMuPDFParser(
            text_kwargs=kwargs.get('text_kwargs'),
            extract_images=kwargs.get('extract_images', False)
        )


for ext, loader_name in DOCUMENT_LOADERS.items():
    DOCUMENT_LOADERS[ext] = globals()[loader_name]


def load_single_document(file_path: Path) -> Document:
    logging.info(f"ATTEMPTING: {file_path.name} ({file_path.suffix})")
    print(f"Processing: {file_path.name}", flush=True)

    file_extension = file_path.suffix.lower()
    loader_class = DOCUMENT_LOADERS.get(file_extension)
    if not loader_class:
        print(f"\033[91mFailed---> {file_path.name} (extension: {file_extension})\033[0m")
        logging.error(f"Unsupported file type: {file_path.name} (extension: {file_extension})")
        return None

    loader_options = {}

    if file_extension in [".epub", ".rtf", ".odt", ".md"]:
        loader_options.update({
            "mode": "single",
            "unstructured_kwargs": {
                "strategy": "fast"
            }
        })
    elif file_extension == ".pdf":
        loader_options.update({
            "extract_images": False,
            "text_kwargs": {},
        })
    elif file_extension in [".eml", ".msg"]:
        loader_options.update({
            "mode": "single",
            "process_attachments": False,
            "unstructured_kwargs": {
                "strategy": "fast"
            }
        })
    elif file_extension == ".html":
        loader_options.update({
            "open_encoding": "utf-8",
            "bs_kwargs": {
                "features": "html.parser",
                "from_encoding": "utf-8",
            },
            "get_text_separator": " ",
        })
    elif file_extension in [".xlsx", ".xls", ".xlsm"]:
        loader_options.update({
            "mode": "single",
            "unstructured_kwargs": {
                "strategy": "fast"
            }
        })
    elif file_extension in [".csv", ".txt"]:
        loader_options.update({
            "encoding": "utf-8",
            "autodetect_encoding": True
        })

    try:
        logging.info(f"Loading with {loader_class.__name__}: {file_path.name}")

        if file_extension in [".epub", ".rtf", ".odt", ".md", ".eml", ".msg", ".xlsx", ".xls", ".xlsm"]:
            unstructured_kwargs = loader_options.pop("unstructured_kwargs", {})
            loader = loader_class(str(file_path), mode=loader_options.get("mode", "single"), **unstructured_kwargs)
        else:
            loader = loader_class(str(file_path), **loader_options)

        logging.info(f"Calling loader.load() for: {file_path.name}")
        documents = loader.load()
        logging.info(f"Loader returned {len(documents) if documents else 0} documents for: {file_path.name}")

        if not documents:
            print(f"\033[91mFailed---> {file_path.name} (No content extracted)\033[0m")
            logging.error(f"No content could be extracted from file: {file_path.name}")
            return None

        document = documents[0]

        content_hash = compute_content_hash(document.page_content)
        metadata = extract_document_metadata(file_path, content_hash)
        document.metadata.update(metadata)
        logging.info(f"SUCCESS: {file_path.name}")
        print(f"Loaded---> {file_path.name}")
        return document

    except (OSError, UnicodeDecodeError) as e:
        print(f"\033[91mFailed---> {file_path.name} (Access/encoding error)\033[0m")
        logging.error(f"File access/encoding error - File: {file_path.name} - Error: {str(e)}")
        return None
    except Exception as e:
        print(f"\033[91mFailed---> {file_path.name} (Unexpected error)\033[0m")
        logging.error(f"Unexpected error processing file: {file_path.name} - Error: {type(e).__name__}: {str(e)}")
        logging.exception("Full traceback:")
        return None


def load_documents(source_dir: Path) -> list:
    valid_extensions = {ext.lower() for ext in DOCUMENT_LOADERS.keys()}
    doc_paths = [f for f in source_dir.iterdir() if f.suffix.lower() in valid_extensions]

    docs = []

    if doc_paths:
        n_workers = min(INGEST_THREADS, max(len(doc_paths), 1))

        executor = None
        try:
            executor = ThreadPoolExecutor(n_workers)
            futures = [executor.submit(load_single_document, path) for path in doc_paths]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        docs.append(result)
                except Exception as e:
                    logging.error(f"Error processing document: {e}")
        except Exception as e:
            logging.error(f"Error in document loading executor: {e}")
            raise
        finally:
            if executor:
                executor.shutdown(wait=True, cancel_futures=True)

    return docs


def add_pymupdf_page_metadata(doc: Document, chunk_size: int = 1200, chunk_overlap: int = 600) -> List[Document]:
    def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int]]:
        if text is None:
            return []

        if isinstance(text, (list, tuple)):
            text = " ".join(str(item) for item in text if item)

        if not isinstance(text, str):
            text = str(text)

        page_markers = []
        offset = 0
        for m in re.finditer(r'\[\[page(\d+)\]\]', text):
            marker_len = len(m.group(0))
            page_markers.append((m.start() - offset, int(m.group(1))))
            offset += marker_len

        clean_text = re.sub(r'\[\[page\d+\]\]', '', text)

        chunks = []
        start = 0
        while start < len(clean_text):
            end = start + chunk_size
            if end > len(clean_text):
                end = len(clean_text)
            chunk = clean_text[start:end].strip()

            page_num = None
            for marker_pos, page in reversed(page_markers):
                if marker_pos <= start:
                    page_num = page
                    break

            if chunk and page_num is not None:
                chunks.append((chunk, page_num))
            elif chunk and page_num is None:
                chunks.append((chunk, 1))

            start += chunk_size - chunk_overlap

        return chunks

    text = doc.page_content

    if text is None:
        logger.warning("Skipping PDF document with None page_content")
        return []

    chunks = split_text(text, chunk_size, chunk_overlap)

    if not chunks:
        logger.warning("No chunks created from PDF document")
        return []

    new_docs = []
    for chunk, page_num in chunks:
        if not chunk or not chunk.strip():
            continue

        new_metadata = doc.metadata.copy() if doc.metadata else {}
        new_metadata['page_number'] = page_num

        new_doc = Document(
            page_content=chunk,
            metadata=new_metadata
        )
        new_docs.append(new_doc)

    return new_docs


def split_documents(documents=None, text_documents_pdf=None):
    try:
        print("\nSplitting documents into chunks.")

        config = get_config()
        chunk_size = config.database.chunk_size
        chunk_overlap = config.database.chunk_overlap

        text_splitter = FixedSizeTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        texts = []

        if documents:
            texts = text_splitter.split_documents(documents)

        if text_documents_pdf:
            processed_pdf_docs = []
            for doc in text_documents_pdf:
                chunked_docs = add_pymupdf_page_metadata(
                    doc,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                processed_pdf_docs.extend(chunked_docs)
            texts.extend(processed_pdf_docs)

        logger.info(f"Total chunks after splitting: {len(texts)}")

        return texts

    except Exception as e:
        logging.exception("Error during document splitting")
        logging.error(f"Error type: {type(e)}")
        raise