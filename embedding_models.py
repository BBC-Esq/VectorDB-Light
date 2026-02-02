import logging
import os
import unicodedata
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

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
            else:
                model_kwargs['attn_implementation'] = 'sdpa'
        else:
            model_kwargs['attn_implementation'] = 'sdpa'
        tokenizer_kwargs = {
            'model_max_length': self.max_seq_length,
        }
        if family == "qwen":
            tokenizer_kwargs['padding_side'] = 'left'
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
        self.model.to(self.device)

    def _safe_encode(self, texts: list[str], batch_size: int = None) -> list[list[float]]:
        bs = batch_size if batch_size is not None else self.batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=bs,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.float().cpu().numpy().tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            embeddings = self._safe_encode(texts, batch_size=self.batch_size)
            return embeddings
        except Exception:
            pass
        all_embeddings = []
        total = len(texts)
        batch_size = self.batch_size
        while batch_size >= 1:
            try:
                for start in range(0, total, batch_size):
                    end = min(start + batch_size, total)
                    batch_texts = texts[start:end]
                    try:
                        batch_embeddings = self._safe_encode(batch_texts, batch_size=batch_size)
                        all_embeddings.extend(batch_embeddings)
                    except Exception:
                        for text in batch_texts:
                            try:
                                single_embedding = self._safe_encode([text], batch_size=1)
                                all_embeddings.extend(single_embedding)
                            except Exception:
                                fallback_text = ''.join(c for c in text if c.isascii() and c.isprintable()).strip() or "empty"
                                single_embedding = self._safe_encode([fallback_text], batch_size=1)
                                all_embeddings.extend(single_embedding)
                return all_embeddings
            except Exception:
                batch_size = batch_size // 2
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        if self.prompt:
            text = self.prompt + text
        if not isinstance(text, str):
            text = str(text)
        text = _normalize_text(text)
        embeddings = self._safe_encode([text], batch_size=1)
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
    return model
