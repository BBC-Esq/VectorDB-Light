# vector_db_creator.py

import faulthandler
faulthandler.enable()

import gc
import logging
import os
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from document_processor import Document
from utilities_core import (
    my_cprint,
    set_cuda_paths,
    configure_logging,
)
from embedding_models import load_embedding_model
from config import get_config
from sqlite_operations import create_metadata_db
from cuda_manager import get_cuda_manager

logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RUST_BACKTRACE", "1")

PROJECT_ROOT = Path(__file__).resolve().parent
STAGE_EXTRACT_PATH = PROJECT_ROOT / "stage_extract.py"
STAGE_SPLIT_PATH = PROJECT_ROOT / "stage_split.py"

from constants import PIPELINE_PRESETS

EXTRACT_MAX_RETRIES = 3
SPLIT_MAX_WORKER_RETRIES = 3
SPLIT_MAX_RETRIES = 5
TILEDB_WRITE_BATCH_SIZE = 100000


def _get_split_params():
    try:
        preset_name = get_config().database.pipeline_preset
    except Exception:
        preset_name = "normal"
    preset = PIPELINE_PRESETS.get(preset_name, PIPELINE_PRESETS["normal"])
    return preset["split_max_parallel_workers"], preset["split_worker_batch_size"]


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


def _run_extract_with_retry(source_dir, output_pkl):
    python = sys.executable
    cmd = [python, str(STAGE_EXTRACT_PATH), str(source_dir), str(output_pkl)]

    for attempt in range(1, EXTRACT_MAX_RETRIES + 1):
        logger.info(f"Extract attempt {attempt}/{EXTRACT_MAX_RETRIES}")
        exit_code, _ = _run_subprocess_stage(f"Extract (attempt {attempt})", cmd)

        if exit_code == 0 and output_pkl.exists():
            logger.info(f"Extract stage completed on attempt {attempt}")
            return

        logger.error(f"Extract attempt {attempt} failed (exit code {exit_code})")

        if attempt < EXTRACT_MAX_RETRIES:
            logger.info("Waiting 3 seconds before retry...")
            time.sleep(3)
            gc.collect()

    raise RuntimeError(f"Extract stage failed after {EXTRACT_MAX_RETRIES} attempts")


def _run_split_with_retry(extracted_pkl, chunks_pkl, chunk_size, chunk_overlap, checkpoint_dir):
    python = sys.executable
    split_parallel, split_batch = _get_split_params()

    for attempt in range(1, SPLIT_MAX_RETRIES + 1):
        logger.info(f"Split attempt {attempt}/{SPLIT_MAX_RETRIES}")

        split_cmd = [
            python, str(STAGE_SPLIT_PATH),
            str(extracted_pkl),
            str(chunks_pkl),
            str(chunk_size),
            str(chunk_overlap),
            "--worker-batch-size", str(split_batch),
            "--max-worker-retries", str(SPLIT_MAX_WORKER_RETRIES),
            "--max-parallel-workers", str(split_parallel),
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
    faulthandler.enable()
    configure_logging("INFO")
    set_cuda_paths()
    _setup_tiledb_dlls()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RUST_BACKTRACE"] = "1"

    create_vector_db = None

    try:
        create_vector_db = CreateVectorDB(database_name=database_name)
        create_vector_db.run()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if create_vector_db:
            del create_vector_db

        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

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

    def _create_tiledb_array(self, texts, vectors_array, metadatas, ids):
        import json

        _setup_tiledb_dlls()

        import tiledb
        import tiledb.vector_search as vs
        from tiledb.vector_search import _tiledbvspy as vspy

        embedding_dim = vectors_array.shape[1]
        num_vectors = vectors_array.shape[0]

        logger.info(f"Creating TileDB array: {num_vectors:,} vectors of dimension {embedding_dim}")
        logger.info(f"Vectors array shape: {vectors_array.shape}, dtype: {vectors_array.dtype}")

        logger.info("Converting IDs to uint64 array...")
        ids_array = np.array([int(id_str) for id_str in ids], dtype=np.uint64)
        logger.info(f"IDs array ready: {ids_array.shape}")

        array_uri = str(self.PERSIST_DIRECTORY / "vectors")

        logger.info("Creating TileDB schema...")
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

        logger.info("Creating TileDB array on disk...")
        tiledb.Array.create(array_uri, schema)
        logger.info("TileDB array schema created.")

        num_batches = (num_vectors + TILEDB_WRITE_BATCH_SIZE - 1) // TILEDB_WRITE_BATCH_SIZE
        logger.info(f"Writing TileDB array in {num_batches} batch(es) "
                     f"of up to {TILEDB_WRITE_BATCH_SIZE:,} records")

        for batch_idx in range(num_batches):
            start = batch_idx * TILEDB_WRITE_BATCH_SIZE
            end = min(start + TILEDB_WRITE_BATCH_SIZE, num_vectors)
            batch_size = end - start

            logger.info(f"  Preparing batch {batch_idx + 1}/{num_batches} "
                        f"(records {start:,}-{end - 1:,})...")

            batch_vectors = vectors_array[start:end]
            batch_ids = ids_array[start:end]
            batch_texts = np.array(texts[start:end], dtype=object)
            batch_metadata = np.array(
                [json.dumps(metadatas[i]) for i in range(start, end)],
                dtype=object
            )

            batch_structured = np.array(
                [tuple(vec) for vec in batch_vectors],
                dtype=[("", np.float32)] * embedding_dim
            )

            logger.info(f"  Writing batch {batch_idx + 1}/{num_batches}...")
            with tiledb.open(array_uri, mode='w') as A:
                A[batch_ids] = {
                    "vector": batch_structured,
                    "text": batch_texts,
                    "metadata": batch_metadata,
                }

            del batch_structured, batch_texts, batch_metadata, batch_vectors
            gc.collect()

            logger.info(f"  Batch {batch_idx + 1}/{num_batches} complete: "
                        f"wrote {batch_size:,} records")

        logger.info("Consolidating TileDB fragments...")
        tiledb.consolidate(array_uri)
        tiledb.vacuum(array_uri)
        logger.info(f"TileDB array consolidated at: {array_uri}")

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
        pipeline_t0 = time.time()

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
            # ============================================================
            # Stage 1: Extract documents via subprocess (with retry)
            # ============================================================
            my_cprint("Extracting documents (subprocess)...", "yellow")
            extract_t0 = time.time()
            _run_extract_with_retry(self.SOURCE_DIRECTORY, extracted_pkl)
            extract_elapsed = time.time() - extract_t0
            logger.info(f"Extract stage: {extract_elapsed:.1f}s")

            with open(extracted_pkl, "rb") as f:
                doc_data = pickle.load(f)
            logger.info(f"Extracted {len(doc_data)} documents")

            if not doc_data:
                my_cprint("No documents found to process.", "red")
                return

            # Build Document objects for metadata DB
            json_docs_to_save = []
            for content, metadata in doc_data:
                json_docs_to_save.append(Document(page_content=content, metadata=metadata))

            del doc_data
            gc.collect()

            # ============================================================
            # Stage 2: Split documents via subprocess (with retry)
            # ============================================================
            my_cprint("Splitting documents into chunks (subprocess)...", "yellow")
            split_t0 = time.time()
            _run_split_with_retry(extracted_pkl, chunks_pkl, chunk_size, chunk_overlap, checkpoint_dir)
            split_elapsed = time.time() - split_t0
            logger.info(f"Split stage: {split_elapsed:.1f}s")

            try:
                extracted_pkl.unlink()
            except Exception:
                pass

            with open(chunks_pkl, "rb") as f:
                split_output = pickle.load(f)

            if isinstance(split_output, dict):
                chunk_texts = split_output["texts"]
                chunks_with_meta = split_output.get("chunks", [])
                del split_output
            else:
                chunk_texts = split_output
                chunks_with_meta = []
                del split_output

            gc.collect()
            logger.info(f"Split into {len(chunk_texts):,} chunks")

            if not chunk_texts:
                my_cprint("No chunks produced after splitting.", "red")
                return

            # ============================================================
            # Build metadata and ID mappings
            # ============================================================
            logger.info("Building metadata and ID mappings...")
            all_metadatas = []
            all_ids = []
            hash_id_mappings = []
            MAX_UINT64 = 18446744073709551615

            logger.info(f"Metadata mapping: {len(chunk_texts):,} chunk_texts, "
                        f"{len(chunks_with_meta):,} chunks_with_meta")
            if chunks_with_meta:
                _, sample_meta = chunks_with_meta[0]
                logger.info(f"Sample metadata keys: {list(sample_meta.keys())}")
            else:
                logger.warning("chunks_with_meta is EMPTY — metadata will be missing!")

            for idx in range(len(chunk_texts)):
                tiledb_id = str(random.randint(0, MAX_UINT64 - 1))

                if idx < len(chunks_with_meta):
                    _, meta = chunks_with_meta[idx]
                else:
                    meta = {}

                file_hash = meta.get('hash', '')
                all_metadatas.append(meta)
                all_ids.append(tiledb_id)
                hash_id_mappings.append((tiledb_id, file_hash))

            logger.info(f"Metadata mapping complete: {len(all_metadatas):,} entries")

            del chunks_with_meta
            gc.collect()

            # ============================================================
            # Stage 3+4: Tokenize + Embed via subprocess pipeline
            # ============================================================
            with cuda_mgr.cuda_operation():
                embeddings = self.initialize_vector_model(EMBEDDING_MODEL_NAME, config_data)

            my_cprint("\nComputing vectors...", "yellow")
            embed_t0 = time.time()

            try:
                self.PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=False)
                my_cprint(f"Created directory: {self.PERSIST_DIRECTORY}", "green")
            except FileExistsError:
                raise FileExistsError(
                    f"Vector database '{self.PERSIST_DIRECTORY.name}' already exists. "
                    "Choose a different name or delete the existing DB first."
                )

            logger.info(f"Total chunks to embed: {len(chunk_texts):,}")

            with cuda_mgr.cuda_operation():
                vectors = embeddings.embed_documents(chunk_texts)

            embed_elapsed = time.time() - embed_t0
            my_cprint(f"Embedding computation completed in {embed_elapsed:.2f} seconds.", "cyan")

            del embeddings
            gc.collect()
            cuda_mgr.force_empty_cache()

            vectors_array = np.ascontiguousarray(vectors, dtype=np.float32)
            del vectors
            gc.collect()

            # ============================================================
            # Stage 5: Write TileDB array + FLAT index (batched)
            # ============================================================
            logger.info("Creating TileDB vector database...")
            try:
                self._create_tiledb_array(chunk_texts, vectors_array, all_metadatas, all_ids)
            except Exception as e:
                logger.error(f"Error creating TileDB database: {e}")
                traceback.print_exc()
                if self.PERSIST_DIRECTORY.exists():
                    try:
                        shutil.rmtree(self.PERSIST_DIRECTORY)
                        logger.info(f"Cleaned up failed database at: {self.PERSIST_DIRECTORY}")
                    except Exception as cleanup_error:
                        logger.error(f"Failed to clean up: {cleanup_error}")
                raise

            my_cprint("Processed all chunks", "yellow")

            pipeline_elapsed = time.time() - pipeline_t0
            my_cprint(f"Database created. Total time: {pipeline_elapsed:.2f} seconds.", "green")

            # ============================================================
            # Stage 6: Write SQLite metadata DB
            # ============================================================
            del chunk_texts, vectors_array, all_metadatas, all_ids
            gc.collect()

            create_metadata_db(self.PERSIST_DIRECTORY, json_docs_to_save, hash_id_mappings)
            del json_docs_to_save, hash_id_mappings
            gc.collect()

            self.clear_docs_for_db_folder()

        except Exception:
            traceback.print_exc()
            raise
        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
