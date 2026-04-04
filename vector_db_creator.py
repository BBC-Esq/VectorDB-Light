# vector_db_creator.py

import gc
import logging
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from collections import defaultdict
import random
import shutil
import traceback
import numpy as np
import torch

from document_processor import Document
from utilities_core import (
    my_cprint,
    set_cuda_paths,
    configure_logging,
)
from embedding_models import load_embedding_model
from embedding_models import create_embedding_model
from config import get_config
from sqlite_operations import create_metadata_db
from cuda_manager import get_cuda_manager

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
STAGE_EXTRACT_PATH = PROJECT_ROOT / "dev" / "stage_extract.py"
STAGE_SPLIT_PATH = PROJECT_ROOT / "dev" / "stage_split.py"

SPLIT_WORKER_BATCH_SIZE = 2000
SPLIT_MAX_WORKER_RETRIES = 3
SPLIT_MAX_PARALLEL_WORKERS = 0
SPLIT_MAX_RETRIES = 5


def _run_subprocess_stage(name, cmd, timeout=3600):
    logger.info(f"Starting subprocess stage: {name}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    output_lines = []
    for line in process.stdout:
        line = line.rstrip("\n")
        if line.strip():
            logger.info(f"  [{name}] {line}")
            output_lines.append(line)

    process.wait(timeout=timeout)

    if process.returncode != 0:
        for line in output_lines[-10:]:
            logger.error(f"  {line}")

    return process.returncode, output_lines


def _run_extract_subprocess(source_dir, output_pkl):
    python = sys.executable
    cmd = [python, str(STAGE_EXTRACT_PATH), str(source_dir), str(output_pkl)]
    exit_code, _ = _run_subprocess_stage("Extract", cmd)
    if exit_code != 0:
        raise RuntimeError(f"Extract stage failed with exit code {exit_code}")


def _run_split_with_retry(extracted_pkl, chunks_pkl, chunk_size, chunk_overlap, checkpoint_dir):
    python = sys.executable

    for attempt in range(1, SPLIT_MAX_RETRIES + 1):
        logger.info(f"Split attempt {attempt}/{SPLIT_MAX_RETRIES}")

        split_cmd = [
            python, str(STAGE_SPLIT_PATH),
            str(extracted_pkl),
            str(chunks_pkl),
            str(chunk_size),
            str(chunk_overlap),
            "--worker-batch-size", str(SPLIT_WORKER_BATCH_SIZE),
            "--max-worker-retries", str(SPLIT_MAX_WORKER_RETRIES),
            "--max-parallel-workers", str(SPLIT_MAX_PARALLEL_WORKERS),
            "--checkpoint-dir", str(checkpoint_dir),
            "--checkpoint-interval", "5",
        ]

        exit_code, _ = _run_subprocess_stage(f"Split (attempt {attempt})", split_cmd)

        if exit_code == 0 and chunks_pkl.exists():
            logger.info(f"Split stage completed on attempt {attempt}")
            return

        logger.error(f"Split attempt {attempt} failed (exit code {exit_code})")

        if attempt < SPLIT_MAX_RETRIES:
            logger.info("Waiting 3 seconds before retry...")
            time.sleep(3)
            gc.collect()

    raise RuntimeError(f"Split stage failed after {SPLIT_MAX_RETRIES} attempts")


def _setup_tiledb_dlls():
    import ctypes
    import tiledb

    venv_root = os.path.dirname(os.path.dirname(sys.executable))
    site_packages = os.path.join(venv_root, 'Lib', 'site-packages')

    tiledb_libs = os.path.join(site_packages, 'tiledb.libs')
    vector_search_lib = os.path.join(site_packages, 'tiledb', 'vector_search', 'lib')

    for directory in [tiledb_libs, vector_search_lib]:
        if os.path.isdir(directory):
            try:
                os.add_dll_directory(directory)
            except OSError:
                pass

    if os.path.isdir(tiledb_libs):
        for filename in sorted(os.listdir(tiledb_libs)):
            if filename.endswith('.dll'):
                try:
                    ctypes.CDLL(os.path.join(tiledb_libs, filename))
                except Exception:
                    pass

    if os.path.isdir(vector_search_lib):
        tiledb_dll = os.path.join(vector_search_lib, 'tiledb.dll')
        if os.path.exists(tiledb_dll):
            try:
                ctypes.CDLL(tiledb_dll)
            except Exception:
                pass


def create_vector_db_in_process(database_name):
    configure_logging("INFO")
    set_cuda_paths()
    _setup_tiledb_dlls()

    embeddings_model = None
    create_vector_db = None

    try:
        create_vector_db = CreateVectorDB(database_name=database_name)
        create_vector_db.run()
    finally:
        if create_vector_db:
            del create_vector_db

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        import time
        time.sleep(0.1)


class CreateVectorDB:
    def __init__(self, database_name):
        self.config = get_config()
        self.SOURCE_DIRECTORY = self.config.docs_dir
        self.PERSIST_DIRECTORY = self.config.vector_db_dir / database_name

    @torch.inference_mode()
    def initialize_vector_model(self, embedding_model_name, config_data):
        return load_embedding_model(
            model_path=embedding_model_name,
            compute_device=config_data.Compute_Device.database_creation,
            use_half=config_data.database.half,
            is_query=False,
            verbose=True,
        )

    @torch.inference_mode()
    def create_database(self, doc_data, chunk_texts, embeddings):
        cuda_mgr = get_cuda_manager()

        my_cprint("\nComputing vectors...", "yellow")
        start_time = time.time()

        hash_id_mappings = []
        MAX_UINT64 = 18446744073709551615

        try:
            self.PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=False)
            my_cprint(f"Created directory: {self.PERSIST_DIRECTORY}", "green")
        except FileExistsError:
            raise FileExistsError(
                f"Vector database '{self.PERSIST_DIRECTORY.name}' already exists. "
                "Choose a different name or delete the existing DB first."
            )

        try:
            all_metadatas = []
            all_ids = []
            chunk_counters = defaultdict(int)

            for idx, text in enumerate(chunk_texts):
                tiledb_id = str(random.randint(0, MAX_UINT64 - 1))

                if idx < len(doc_data) and doc_data:
                    _, meta = doc_data[idx]
                else:
                    meta = {}

                file_hash = meta.get('hash', '')
                chunk_counters[file_hash] = chunk_counters.get(file_hash, 0) + 1
                all_metadatas.append(meta)
                all_ids.append(tiledb_id)
                hash_id_mappings.append((tiledb_id, file_hash))

            logger.info(f"Total chunks to embed: {len(chunk_texts)}")

            embedding_start_time = time.time()

            with cuda_mgr.cuda_operation():
                vectors = embeddings.embed_documents(chunk_texts)

            embedding_end_time = time.time()
            embedding_elapsed = embedding_end_time - embedding_start_time
            my_cprint(f"Embedding computation completed in {embedding_elapsed:.2f} seconds.", "cyan")

            vectors_array = np.ascontiguousarray(vectors, dtype=np.float32)

            logger.info("Creating TileDB vector database...")
            self._create_tiledb_array(chunk_texts, vectors_array, all_metadatas, all_ids)

            my_cprint("Processed all chunks", "yellow")

            end_time = time.time()
            elapsed_time = end_time - start_time
            my_cprint(f"Database created. Elapsed time: {elapsed_time:.2f} seconds.", "green")

            return hash_id_mappings

        except Exception as e:
            logger.error(f"Error creating database '{self.PERSIST_DIRECTORY.name}': {str(e)}")
            logger.error(f"Processing {len(chunk_texts) if chunk_texts else 0} chunks when error occurred")
            traceback.print_exc()
            if self.PERSIST_DIRECTORY.exists():
                try:
                    shutil.rmtree(self.PERSIST_DIRECTORY)
                    logger.info(f"Cleaned up failed database creation at: {self.PERSIST_DIRECTORY}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up database directory: {cleanup_error}")
            raise

    def _create_tiledb_array(self, texts, vectors, metadatas, ids):
        import json

        _setup_tiledb_dlls()

        import tiledb
        import tiledb.vector_search as vs
        from tiledb.vector_search import _tiledbvspy as vspy

        embedding_dim = len(vectors[0])
        num_vectors = len(vectors)

        logger.info(f"Creating TileDB array: {num_vectors} vectors of dimension {embedding_dim}")

        vectors_array = np.array(vectors, dtype=np.float32)
        if vectors_array.ndim == 1:
            vectors_array = vectors_array.reshape(num_vectors, embedding_dim)
        vectors_array = np.ascontiguousarray(vectors_array)

        logger.info(f"Vectors array shape: {vectors_array.shape}, dtype: {vectors_array.dtype}")

        ids_array = np.array([int(id_str) for id_str in ids], dtype=np.uint64)
        texts_array = np.array(texts, dtype=object)

        metadata_strings = [json.dumps(meta) for meta in metadatas]
        metadata_array = np.array(metadata_strings, dtype=object)

        array_uri = str(self.PERSIST_DIRECTORY / "vectors")

        dom = tiledb.Domain(
            tiledb.Dim(name="id", domain=(0, np.iinfo(np.uint64).max - 20000), tile=10000, dtype=np.uint64)
        )

        attrs = [
            tiledb.Attr(name="vector", dtype=np.dtype([("", np.float32)] * embedding_dim)),
            tiledb.Attr(name="text", dtype=str, var=True),
            tiledb.Attr(name="metadata", dtype=str, var=True),
        ]

        schema = tiledb.ArraySchema(
            domain=dom,
            attrs=attrs,
            sparse=True,
            cell_order='row-major',
            tile_order='row-major'
        )

        tiledb.Array.create(array_uri, schema)

        vectors_structured = np.array([tuple(vec) for vec in vectors_array],
                                      dtype=[("", np.float32)] * embedding_dim)

        with tiledb.open(array_uri, mode='w') as A:
            A[ids_array] = {
                "vector": vectors_structured,
                "text": texts_array,
                "metadata": metadata_array
            }

        logger.info(f"TileDB array created at: {array_uri}")

        logger.info("Creating TileDB FLAT vector search index via ingest...")
        index_uri = str(self.PERSIST_DIRECTORY / "vector_index")

        index = vs.ingest(
            index_type="FLAT",
            index_uri=index_uri,
            input_vectors=vectors_array,
            external_ids=ids_array,
            dimensions=embedding_dim,
            distance_metric=vspy.DistanceMetric.COSINE
        )

        metadata_file = self.PERSIST_DIRECTORY / "index_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'distance_metric': 'cosine',
                'dimensions': embedding_dim,
                'vector_type': 'float32',
                'index_type': 'FLAT',
                'num_vectors': num_vectors
            }, f)

        logger.info(f"FLAT index created at: {index_uri}")

    def clear_docs_for_db_folder(self):
        for item in self.SOURCE_DIRECTORY.iterdir():
            if item.is_file() or item.is_symlink():
                try:
                    item.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete {item}: {e}")

    @torch.inference_mode()
    def run(self):
        cuda_mgr = get_cuda_manager()

        config_data = get_config()
        EMBEDDING_MODEL_NAME = config_data.EMBEDDING_MODEL_NAME
        chunk_size = config_data.database.chunk_size
        chunk_overlap = config_data.database.chunk_overlap

        tmp_dir = tempfile.mkdtemp(prefix="vectordb_create_")
        tmp_path = Path(tmp_dir)
        extracted_pkl = tmp_path / "extracted.pkl"
        chunks_pkl = tmp_path / "chunks.pkl"
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        try:
            # Stage 1: Extract documents via subprocess
            my_cprint("Extracting documents (subprocess)...", "yellow")
            _run_extract_subprocess(self.SOURCE_DIRECTORY, extracted_pkl)

            with open(extracted_pkl, "rb") as f:
                doc_data = pickle.load(f)
            logger.info(f"Extracted {len(doc_data)} documents")

            if not doc_data:
                my_cprint("No documents found to process.", "red")
                return

            # Build Document objects for metadata DB (lightweight, no processing)
            json_docs_to_save = []
            for content, metadata in doc_data:
                json_docs_to_save.append(Document(page_content=content, metadata=metadata))

            # Stage 2: Split documents via subprocess
            my_cprint("Splitting documents into chunks (subprocess)...", "yellow")
            _run_split_with_retry(extracted_pkl, chunks_pkl, chunk_size, chunk_overlap, checkpoint_dir)

            with open(chunks_pkl, "rb") as f:
                split_output = pickle.load(f)

            if isinstance(split_output, dict):
                chunk_texts = split_output["texts"]
                chunks_with_meta = split_output.get("chunks", [])
            else:
                chunk_texts = split_output
                chunks_with_meta = []

            logger.info(f"Split into {len(chunk_texts)} chunks")

            if not chunk_texts:
                my_cprint("No chunks produced after splitting.", "red")
                return

            # Stage 3+4: Tokenize + embed via subprocess tokenization pipeline
            with cuda_mgr.cuda_operation():
                embeddings = self.initialize_vector_model(EMBEDDING_MODEL_NAME, config_data)

            hash_id_mappings = self.create_database(chunks_with_meta, chunk_texts, embeddings)

            del chunk_texts, embeddings
            gc.collect()

            cuda_mgr.force_empty_cache()

            create_metadata_db(self.PERSIST_DIRECTORY, json_docs_to_save, hash_id_mappings)
            del json_docs_to_save
            gc.collect()
            self.clear_docs_for_db_folder()

        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
