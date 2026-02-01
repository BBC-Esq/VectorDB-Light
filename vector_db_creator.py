# vector_db_creator.py

import gc
import logging
import os
import time
from pathlib import Path
from collections import defaultdict
import random
import shutil
import traceback
import torch
import tiledb
import numpy as np
import tiledb.vector_search as vs
from tiledb.vector_search import _tiledbvspy as vspy

from document_processor import load_documents, split_documents
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


def create_vector_db_in_process(database_name):
    configure_logging("INFO")
    set_cuda_paths()

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
    def create_database(self, texts, embeddings):
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
            all_texts = []
            all_metadatas = []
            all_ids = []
            chunk_counters = defaultdict(int)

            for idx, doc in enumerate(texts):
                file_hash = doc.metadata.get('hash')
                chunk_counters[file_hash] += 1
                tiledb_id = str(random.randint(0, MAX_UINT64 - 1))

                all_texts.append(doc.page_content)
                all_metadatas.append(doc.metadata)
                all_ids.append(tiledb_id)
                hash_id_mappings.append((tiledb_id, file_hash))

            logger.info(f"Total chunks to embed: {len(all_texts)}")

            logger.info("Cleaning text before embedding...")
            cleaned_texts = []
            cleaned_metadatas = []
            cleaned_ids = []
            cleaned_hash_mappings = []

            for i, text in enumerate(all_texts):
                if text is None:
                    logger.warning(f"Skipping None text at index {i}")
                    continue
                if isinstance(text, (list, tuple)):
                    text = " ".join(str(item) for item in text if item)
                    logger.warning(f"Flattened list/tuple to string at index {i}")
                if not isinstance(text, str):
                    text = str(text)

                cleaned = "".join(
                    char for char in text
                    if char == "\n" or char == "\t" or (ord(char) >= 32 and ord(char) != 127)
                ).strip()

                if not cleaned:
                    logger.warning(f"Skipping empty text after cleaning at index {i}")
                    continue

                cleaned_texts.append(cleaned)
                cleaned_metadatas.append(all_metadatas[i])
                cleaned_ids.append(all_ids[i])
                cleaned_hash_mappings.append(hash_id_mappings[i])

            all_texts = cleaned_texts
            all_metadatas = cleaned_metadatas
            all_ids = cleaned_ids
            hash_id_mappings = cleaned_hash_mappings

            logger.info(f"Text cleaning completed. {len(all_texts)} chunks remaining.")

            embedding_start_time = time.time()

            with cuda_mgr.cuda_operation():
                vectors = embeddings.embed_documents(all_texts)

            embedding_end_time = time.time()
            embedding_elapsed = embedding_end_time - embedding_start_time
            my_cprint(f"Embedding computation completed in {embedding_elapsed:.2f} seconds.", "cyan")

            logger.info("Creating TileDB vector database...")
            self._create_tiledb_array(all_texts, vectors, all_metadatas, all_ids)

            my_cprint("Processed all chunks", "yellow")

            end_time = time.time()
            elapsed_time = end_time - start_time
            my_cprint(f"Database created. Elapsed time: {elapsed_time:.2f} seconds.", "green")

            return hash_id_mappings

        except Exception as e:
            logger.error(f"Error creating database '{self.PERSIST_DIRECTORY.name}': {str(e)}")
            logger.error(f"Processing {len(all_texts) if 'all_texts' in locals() else 0} chunks when error occurred")
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

        documents = []

        text_documents = load_documents(self.SOURCE_DIRECTORY)
        if isinstance(text_documents, list) and text_documents:
            documents.extend(text_documents)

        text_documents_pdf = [doc for doc in documents if doc.metadata.get("file_type") == ".pdf"]
        documents = [doc for doc in documents if doc.metadata.get("file_type") != ".pdf"]

        json_docs_to_save = []
        json_docs_to_save.extend(documents)
        json_docs_to_save.extend(text_documents_pdf)

        texts = []

        if (isinstance(documents, list) and documents) or (isinstance(text_documents_pdf, list) and text_documents_pdf):
            texts = split_documents(documents, text_documents_pdf)
            logger.info(f"Documents split into {len(texts)} chunks.")

        del documents, text_documents_pdf
        gc.collect()

        if isinstance(texts, list) and texts:
            with cuda_mgr.cuda_operation():
                embeddings = self.initialize_vector_model(EMBEDDING_MODEL_NAME, config_data)

            hash_id_mappings = self.create_database(texts, embeddings)

            del texts, embeddings
            gc.collect()

            cuda_mgr.force_empty_cache()

            create_metadata_db(self.PERSIST_DIRECTORY, json_docs_to_save, hash_id_mappings)
            del json_docs_to_save
            gc.collect()
            self.clear_docs_for_db_folder()