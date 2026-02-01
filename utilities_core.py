# utilities_core.py
import importlib
import importlib.metadata
import importlib.util
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from termcolor import cprint
import psutil
import torch
from packaging import version

from config import get_config
from constants import VECTOR_MODELS

logger = logging.getLogger(__name__)


def set_cuda_paths():
    if platform.system() != "Windows":
        logger.debug("Skipping CUDA path setup: not on Windows")
        return

    venv_base = Path(sys.executable).parent.parent
    nvidia_base = venv_base / 'Lib' / 'site-packages' / 'nvidia'

    if not nvidia_base.exists():
        logger.debug("Skipping CUDA path setup: nvidia packages not found")
        return

    logger.info("Setting up CUDA paths for Windows")

    cuda_path_runtime = nvidia_base / 'cuda_runtime' / 'bin'
    cuda_path_runtime_lib = nvidia_base / 'cuda_runtime' / 'lib' / 'x64'
    cuda_path_runtime_include = nvidia_base / 'cuda_runtime' / 'include'
    cublas_path = nvidia_base / 'cublas' / 'bin'
    cudnn_path = nvidia_base / 'cudnn' / 'bin'
    nvrtc_path = nvidia_base / 'cuda_nvrtc' / 'bin'
    nvcc_path = nvidia_base / 'cuda_nvcc' / 'bin'

    paths_to_add = [
        cuda_path_runtime,
        cuda_path_runtime_lib,
        cuda_path_runtime_include,
        cublas_path,
        cudnn_path,
        nvrtc_path,
        nvcc_path,
    ]

    # Add to PATH (still useful for some tools and subprocesses)
    current_value = os.environ.get('PATH', '')
    new_value = os.pathsep.join([str(p) for p in paths_to_add] + ([current_value] if current_value else []))
    os.environ['PATH'] = new_value

    # Set CUDA_PATH (used by some frameworks)
    triton_cuda_path = nvidia_base / 'cuda_runtime'
    current_cuda_path = os.environ.get('CUDA_PATH', '')
    new_cuda_path = os.pathsep.join([str(triton_cuda_path)] + ([current_cuda_path] if current_cuda_path else []))
    os.environ['CUDA_PATH'] = new_cuda_path

    # Required for Python 3.8+ on Windows: register DLL directories
    if hasattr(os, 'add_dll_directory'):
        for path in paths_to_add:
            if path.exists():
                try:
                    os.add_dll_directory(str(path))
                    logger.debug(f"Added DLL directory: {path}")
                except OSError as e:
                    logger.warning(f"Failed to add DLL directory {path}: {e}")

    logger.info("CUDA paths configured successfully")


def clean_triton_cache():
    triton_cache_dir = Path.home() / '.triton'

    if triton_cache_dir.exists():
        try:
            logger.info(f"Removing Triton cache at {triton_cache_dir}")
            shutil.rmtree(triton_cache_dir)
            logger.info("Triton cache successfully removed")
            return True
        except Exception as e:
            logger.warning(f"Failed to remove Triton cache: {e}")
            return False
    else:
        logger.debug("No Triton cache found to clean")
        return True


def supports_flash_attention():
    if not torch.cuda.is_available():
        logger.debug("CUDA not available, flash attention not supported")
        return False

    major, minor = torch.cuda.get_device_capability()
    supports = major >= 8
    logger.debug(f"CUDA compute capability: {major}.{minor}, flash attention {'supported' if supports else 'not supported'}")
    return supports

def get_model_native_precision(embedding_model_name: str) -> str:
    logger.debug(f"Looking for precision for model: {embedding_model_name}")
    model_name = os.path.basename(embedding_model_name)
    repo_style_name = model_name.replace('--', '/')

    for group_name, group_models in VECTOR_MODELS.items():
        for model in group_models:
            if model['repo_id'] == repo_style_name or model['name'] in model_name:
                logger.debug(f"Found match in {group_name}! Using precision: {model['precision']}")
                return model['precision']

    logger.debug("No match found, defaulting to float32")
    return 'float32'


def get_appropriate_dtype(compute_device: str, use_half: bool, model_native_precision: str) -> torch.dtype:
    logger.debug(f"Computing dtype - device: {compute_device}, use_half: {use_half}, native: {model_native_precision}")

    compute_device = compute_device.lower()
    model_native_precision = model_native_precision.lower()

    if compute_device == 'cpu':
        logger.debug("Using CPU, returning float32")
        return torch.float32

    cuda_available = torch.cuda.is_available()
    cuda_capability = torch.cuda.get_device_capability() if cuda_available else (0, 0)

    if model_native_precision == 'bfloat16':
        if use_half:
            if cuda_available and cuda_capability[0] >= 8:
                logger.debug("Model native bfloat16, GPU supports it, returning bfloat16")
                return torch.bfloat16
            else:
                logger.debug("GPU doesn't support bfloat16, falling back to float16")
                return torch.float16
        else:
            logger.debug("Half checkbox not checked for bfloat16 model, returning float32")
            return torch.float32

    elif model_native_precision == 'float16':
        if use_half and cuda_available:
            logger.debug("Model native float16 and CUDA available, returning float16")
            return torch.float16
        else:
            logger.debug("Returning float32 for float16 model")
            return torch.float32

    elif model_native_precision == 'float32':
        if not use_half:
            logger.debug("Model is float32 and use_half is False, returning float32")
            return torch.float32
        else:
            if cuda_available:
                if cuda_capability[0] >= 8:
                    logger.debug("Using bfloat16 due to Ampere+ GPU")
                    return torch.bfloat16
                else:
                    logger.debug("Using float16 due to pre-Ampere GPU")
                    return torch.float16
            else:
                logger.debug("No CUDA available, returning float32")
                return torch.float32

    logger.debug(f"Unrecognized precision '{model_native_precision}', returning float32")
    return torch.float32


def get_embedding_batch_size(model_name: str, compute_device: str) -> int:
    if compute_device.lower() == 'cpu':
        return 2

    batch_size_mapping = {
        'Qwen3-Embedding-0.6B': 4,
        'bge-small-en-v1.5': 12,
        'bge-base-en-v1.5': 8,
        'bge-large-en-v1.5': 6,
    }

    model_name_lower = model_name.lower()
    for key, value in batch_size_mapping.items():
        if key.lower() in model_name_lower:
            logger.debug(f"Using batch size {value} for {key}")
            return value

    logger.debug("Using default batch size 8")
    return 8


def my_cprint(message: str, color: str = "white"):
    try:
        cprint(message, color, flush=True)
    except Exception:
        print(message, flush=True)

def get_embedding_dtype_and_batch(
    compute_device: str,
    use_half: bool,
    model_native_precision: str,
    model_name: str,
    is_query: bool,
):
    dtype = get_appropriate_dtype(compute_device, use_half, model_native_precision)
    batch = 1 if is_query else get_embedding_batch_size(model_name, compute_device)
    return dtype, batch

def configure_logging(level: str = "INFO"):
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return
    root.setLevel(level.upper())
    h = logging.StreamHandler()
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    h.setFormatter(fmt)
    root.addHandler(h)