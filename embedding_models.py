import logging
import os
import gc
import unicodedata
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import get_config
from utilities_core import (
    supports_flash_attention,
    get_embedding_dtype_and_batch,
    get_model_native_precision
)

logger = logging.getLogger(__name__)


def _get_model_family(model_path: str) -> str:
    model_path_lower = model_path.lower()
    if "qwen" in model_path_lower or "qwen3-embedding" in model_path_lower:
        return "qwen"
    elif "bge" in model_path_lower:
        return "bge"
    else:
        return "generic"


def _get_prompt_for_family(family: str, is_query: bool = False) -> str:
    if family == "qwen" and is_query:
        return "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
    elif family == "bge":
        return "Represent this sentence for searching relevant passages: "
    else:
        return ""


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)

    cleaned = []
    for char in text:
        if char in '\n\t\r':
            cleaned.append(' ')
        elif ord(char) < 32:
            continue
        elif ord(char) == 127:
            continue
        elif ord(char) > 65535:
            continue
        else:
            cleaned.append(char)

    result = ''.join(cleaned)
    result = ' '.join(result.split())

    return result.strip() or " "


def _validate_and_clean_texts(texts: list) -> list[str]:
    cleaned = []

    for idx, text in enumerate(texts):
        if text is None:
            cleaned.append(" ")
            continue

        if isinstance(text, (list, tuple)):
            parts = []
            for item in text:
                if item is not None:
                    parts.append(str(item))
            text = " ".join(parts) if parts else " "

        if not isinstance(text, str):
            text = str(text)

        text = _normalize_text(text)
        cleaned.append(text)

    return cleaned


class DirectEmbeddingModel:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: torch.dtype = None,
        batch_size: int = 8,
        max_seq_length: int = 512,
        prompt: str = "",
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.prompt = prompt
        self.model = None
        self.tokenizer = None

        logger.info(f"Initializing DirectEmbeddingModel: {os.path.basename(model_path)}")
        self._initialize_model()

    def _initialize_model(self):
        family = _get_model_family(self.model_path)

        model_kwargs = {
            'torch_dtype': self.dtype if self.dtype else torch.float32,
        }

        is_cuda = self.device.lower().startswith("cuda")
        if family == "qwen":
            if is_cuda and supports_flash_attention():
                model_kwargs['attn_implementation'] = 'flash_attention_2'
                logger.debug("Using flash_attention_2 for Qwen model")
            else:
                model_kwargs['attn_implementation'] = 'sdpa'
                logger.debug("Using sdpa for Qwen model")
        else:
            model_kwargs['attn_implementation'] = 'sdpa'

        tokenizer_kwargs = {
            'model_max_length': self.max_seq_length,
        }

        if family == "qwen":
            tokenizer_kwargs['padding_side'] = 'left'

        logger.info("Loading SentenceTransformer model...")
        self.model = SentenceTransformer(
            model_name_or_path=self.model_path,
            device=self.device,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )

        self.model.max_seq_length = self.max_seq_length

        if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
            self.tokenizer = self.model.tokenizer

            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            logger.info(f"Tokenizer pad_token: {self.tokenizer.pad_token}")

        self.model.to(self.device)

        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"  - Dtype: {self.dtype}")
        logger.info(f"  - Batch size: {self.batch_size}")
        logger.info(f"  - Max sequence length: {self.max_seq_length}")

    # # THIS IS CAUTIOUS AND AVOIDS BATCHING, DESTROYS NSIGHT, BUT IS SAFE
    # def _safe_encode(self, texts: list[str]) -> list[list[float]]:
        # features = self.tokenizer(
            # texts,
            # padding=True,
            # truncation=True,
            # max_length=self.max_seq_length,
            # return_tensors='pt'
        # )

        # features = {k: v.to(self.device) for k, v in features.items()}

        # with torch.no_grad():
            # sentence_features = features

            # for module in self.model._modules.values():
                # sentence_features = module(sentence_features)

            # embeddings = sentence_features['sentence_embedding']
            # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # # this is commented out because the embeddings must be in float32 for storage
        # # return embeddings.cpu().numpy().tolist()
        # return embeddings.float().cpu().numpy().tolist() # this one converts to float32

    # THIS REIMPLEMENTS BATCHING FOLLOWING NSIGHT'S BAD REPORT
    def _safe_encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.float().cpu().numpy().tolist()


    # interval based memory clear
    # def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # if not texts:
            # return []

        # texts = _validate_and_clean_texts(texts)
        
        # logger.info(f"Embedding {len(texts)} documents...")

        # all_embeddings = []
        # failed_batches = []
        # total = len(texts)
        
        # for batch_start in tqdm(range(0, total, self.batch_size), desc="Batches", unit="batch"):
            # batch_end = min(batch_start + self.batch_size, total)
            # batch_texts = texts[batch_start:batch_end]
            
            # try:
                # batch_embeddings = self._safe_encode(batch_texts)
                # all_embeddings.extend(batch_embeddings)
                
            # except Exception as e:
                # logger.error(f"Batch {batch_start}-{batch_end} failed: {e}")

                # # Enhanced diagnostics - tokenize individually to analyze
                # token_info = []
                # for text in batch_texts:
                    # try:
                        # # Tokenize individually (this always works)
                        # individual_tokens = self.tokenizer(
                            # text,
                            # padding=False,
                            # truncation=True,
                            # max_length=self.max_seq_length,
                            # return_tensors=None
                        # )
                        # token_info.append({
                            # 'token_count': len(individual_tokens['input_ids']),
                            # 'first_10_tokens': individual_tokens['input_ids'][:10],
                            # 'last_10_tokens': individual_tokens['input_ids'][-10:],
                            # 'has_special_tokens': self.tokenizer.cls_token_id in individual_tokens['input_ids'],
                        # })
                    # except Exception as tok_err:
                        # token_info.append({
                            # 'token_count': -1,
                            # 'tokenization_error': str(tok_err)
                        # })

                # token_counts = [t['token_count'] for t in token_info if t['token_count'] > 0]

                # failed_batch_info = {
                    # 'batch_start': batch_start,
                    # 'batch_end': batch_end,
                    # 'texts': batch_texts,
                    # 'text_lengths': [len(t) for t in batch_texts],
                    # 'token_counts': token_counts,
                    # 'token_count_min': min(token_counts) if token_counts else None,
                    # 'token_count_max': max(token_counts) if token_counts else None,
                    # 'token_count_variance': max(token_counts) - min(token_counts) if token_counts else None,
                    # 'token_details': token_info,
                    # 'error': str(e)
                # }
                # failed_batches.append(failed_batch_info)

                # # Continue with fallback processing
                # for idx, text in enumerate(batch_texts):
                    # global_idx = batch_start + idx
                    # try:
                        # single_embedding = self._safe_encode([text])
                        # all_embeddings.extend(single_embedding)
                    # except Exception as e_single:
                        # logger.error(f"Single encode failed at {global_idx}: {e_single}")
                        # try:
                            # fallback_text = ''.join(c for c in text if c.isascii() and c.isprintable()).strip() or "empty"
                            # single_embedding = self._safe_encode([fallback_text])
                            # all_embeddings.extend(single_embedding)
                        # except Exception:
                            # placeholder = self._safe_encode(["placeholder"])
                            # all_embeddings.extend(placeholder)
                            # logger.error(f"Using placeholder for index {global_idx}")

            # if batch_start > 0 and batch_start % (self.batch_size * 500) == 0:
                # gc.collect()
                # if torch.cuda.is_available():
                    # torch.cuda.empty_cache()

        # if failed_batches:
            # import json
            # from pathlib import Path
            # failed_batches_path = Path(__file__).parent / "failed_batches.json"
            # with open(failed_batches_path, 'w', encoding='utf-8') as f:
                # json.dump(failed_batches, f, indent=2, ensure_ascii=False)
            # logger.info(f"Saved {len(failed_batches)} failed batches to {failed_batches_path}")

        # logger.info(f"Embedding complete. Generated {len(all_embeddings)} embeddings.")
        
        # return all_embeddings

    # only clear memory upon batch fail
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # second instance of cleaning texts after vector_db_creator.py does it
        # texts = _validate_and_clean_texts(texts)

        # one extra cache clear, not in above method, trying here first
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Embedding {len(texts)} documents...")

        all_embeddings = []
        failed_batches = []
        total = len(texts)
        
        for batch_start in tqdm(range(0, total, self.batch_size), desc="Batches", unit="batch"):
            batch_end = min(batch_start + self.batch_size, total)
            batch_texts = texts[batch_start:batch_end]
            
            try:
                batch_embeddings = self._safe_encode(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Batch {batch_start}-{batch_end} failed: {e}")
                
                token_info = []
                for text in batch_texts:
                    try:
                        individual_tokens = self.tokenizer(
                            text,
                            padding=False,
                            truncation=True,
                            max_length=self.max_seq_length,
                            return_tensors=None
                        )
                        token_info.append({
                            'token_count': len(individual_tokens['input_ids']),
                            'first_10_tokens': individual_tokens['input_ids'][:10],
                            'last_10_tokens': individual_tokens['input_ids'][-10:],
                            'has_special_tokens': self.tokenizer.cls_token_id in individual_tokens['input_ids'],
                        })
                    except Exception as tok_err:
                        token_info.append({
                            'token_count': -1,
                            'tokenization_error': str(tok_err)
                        })
                
                token_counts = [t['token_count'] for t in token_info if t['token_count'] > 0]
                
                failed_batch_info = {
                    'batch_start': batch_start,
                    'batch_end': batch_end,
                    'texts': batch_texts,
                    'text_lengths': [len(t) for t in batch_texts],
                    'token_counts': token_counts,
                    'token_count_min': min(token_counts) if token_counts else None,
                    'token_count_max': max(token_counts) if token_counts else None,
                    'token_count_variance': max(token_counts) - min(token_counts) if token_counts else None,
                    'token_details': token_info,
                    'error': str(e)
                }
                failed_batches.append(failed_batch_info)
                
                for idx, text in enumerate(batch_texts):
                    global_idx = batch_start + idx
                    try:
                        single_embedding = self._safe_encode([text])
                        all_embeddings.extend(single_embedding)
                    except Exception as e_single:
                        logger.error(f"Single encode failed at {global_idx}: {e_single}")
                        try:
                            fallback_text = ''.join(c for c in text if c.isascii() and c.isprintable()).strip() or "empty"
                            single_embedding = self._safe_encode([fallback_text])
                            all_embeddings.extend(single_embedding)
                        except Exception:
                            placeholder = self._safe_encode(["placeholder"])
                            all_embeddings.extend(placeholder)
                            logger.error(f"Using placeholder for index {global_idx}")
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if failed_batches:
            import json
            from pathlib import Path
            failed_batches_path = Path(__file__).parent / "failed_batches.json"
            with open(failed_batches_path, 'w', encoding='utf-8') as f:
                json.dump(failed_batches, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(failed_batches)} failed batches to {failed_batches_path}")

        logger.info(f"Embedding complete. Generated {len(all_embeddings)} embeddings.")
        
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        if self.prompt:
            text = self.prompt + text

        if not isinstance(text, str):
            text = str(text)
        
        text = _normalize_text(text)

        embeddings = self._safe_encode([text])
        return embeddings[0]

    def __del__(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None


def create_embedding_model(
    model_path: str,
    compute_device: str = "cpu",
    dtype: torch.dtype = None,
    batch_size: int = None,
    is_query: bool = False,
) -> DirectEmbeddingModel:

    config = get_config()
    model_name = os.path.basename(model_path)

    family = _get_model_family(model_path)
    model_native_precision = get_model_native_precision(model_name)

    use_half = config.database.half
    _dtype, _batch_size = get_embedding_dtype_and_batch(
        compute_device=compute_device,
        use_half=use_half,
        model_native_precision=model_native_precision,
        model_name=model_name,
        is_query=is_query
    )

    final_dtype = dtype if dtype is not None else _dtype
    final_batch_size = batch_size if batch_size is not None else _batch_size

    if family == "qwen":
        max_seq_length = 8192
    else:
        max_seq_length = 512

    prompt = _get_prompt_for_family(family, is_query)

    logger.info(f"Creating embedding model: {model_name}")
    logger.info(f"  - Family: {family}")
    logger.info(f"  - Device: {compute_device}")
    logger.info(f"  - Dtype: {final_dtype}")
    logger.info(f"  - Batch size: {final_batch_size}")
    logger.info(f"  - Max sequence: {max_seq_length}")
    if prompt:
        logger.info(f"  - Using prompt: {prompt[:50]}...")

    return DirectEmbeddingModel(
        model_path=model_path,
        device=compute_device,
        dtype=final_dtype,
        batch_size=final_batch_size,
        max_seq_length=max_seq_length,
        prompt=prompt,
    )


def load_embedding_model(
    model_path: str,
    compute_device: str,
    use_half: bool,
    is_query: bool = False,
    verbose: bool = False,
) -> DirectEmbeddingModel:

    model_name = os.path.basename(model_path)
    model_native_precision = get_model_native_precision(model_name)

    dtype, batch_size = get_embedding_dtype_and_batch(
        compute_device=compute_device,
        use_half=use_half,
        model_native_precision=model_native_precision,
        model_name=model_name,
        is_query=is_query,
    )

    model = create_embedding_model(
        model_path=model_path,
        compute_device=compute_device,
        dtype=dtype,
        batch_size=batch_size,
        is_query=is_query,
    )

    if verbose:
        from utilities_core import my_cprint
        precision = "float32" if dtype is None else str(dtype).split('.')[-1]
        my_cprint(f"{model_name} ({precision}) loaded using a batch size of {batch_size}.", "green")

    return model