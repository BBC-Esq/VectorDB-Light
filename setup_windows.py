"""
Windows Installation Script
Uses uv for fast, reliable dependency resolution.
"""

import os
import subprocess
import sys
import time
import shutil
from pathlib import Path

# Enable ANSI colors on Windows
os.system("")


class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"


def print_success(msg):
    print(f"{Colors.GREEN}{msg}{Colors.RESET}")


def print_warning(msg):
    print(f"{Colors.YELLOW}{msg}{Colors.RESET}")


def print_error(msg):
    print(f"{Colors.RED}{msg}{Colors.RESET}")


def print_header(msg):
    print(f"\n{'='*60}")
    print(f" {msg}")
    print(f"{'='*60}\n")


def run_command(cmd, description=None, check=True, timeout=600):
    """Run a command with proper error handling."""
    if description:
        print(f"\n{description}...")
    
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd)}")
        print_error(f"Error: {e.stderr.strip()}")
        return False, e.stderr
    except subprocess.TimeoutExpired:
        print_error(f"Command timed out after {timeout}s")
        return False, "Timeout"


def has_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def show_message_box(title, message, msg_type="info", yes_no=False):
    """Show a tkinter message box."""
    import tkinter as tk
    from tkinter import messagebox
    
    root = tk.Tk()
    root.withdraw()
    
    if yes_no:
        result = messagebox.askyesno(title, message)
    elif msg_type == "error":
        messagebox.showerror(title, message)
        result = False
    else:
        messagebox.showinfo(title, message)
        result = True
    
    root.destroy()
    return result


def check_prerequisites():
    """Check Python version and prerequisites."""
    major, minor = sys.version_info.major, sys.version_info.minor
    
    if not (major == 3 and minor in [11, 12, 13]):
        show_message_box(
            "Python Version Error",
            f"Python 3.11, 3.12, or 3.13 is required.\n\nDetected: Python {major}.{minor}",
            msg_type="error"
        )
        return False
    
    if not show_message_box(
        "Confirmation",
        f"Python {major}.{minor} detected (compatible).\n\nClick YES to proceed or NO to exit.",
        yes_no=True
    ):
        return False
    
    # Check for NVIDIA GPU
    has_gpu = has_nvidia_gpu()
    if has_gpu:
        msg = "NVIDIA GPU detected.\n\nProceed with GPU installation?"
    else:
        msg = "No NVIDIA GPU detected. GPU is required for full functionality.\n\nProceed anyway?"
    
    if not show_message_box("GPU Detection", msg, yes_no=True):
        return False
    
    # Check manual prerequisites
    prerequisites = [
        ("Git", "Have you installed Git?"),
        ("Git LFS", "Have you installed Git Large File Storage?"),
        ("Pandoc", "Have you installed Pandoc?"),
        ("Build Tools", "Have you installed Microsoft Build Tools / Visual Studio with C++ libraries?"),
    ]
    
    for name, question in prerequisites:
        if not show_message_box("Prerequisite Check", question, yes_no=True):
            print_error(f"Installation cancelled: {name} not confirmed.")
            return False
    
    return True


def clean_triton_cache():
    """Remove Triton cache directory."""
    cache_dir = Path.home() / ".triton"
    
    if cache_dir.is_dir():
        print(f"Removing Triton cache at {cache_dir}...")
        try:
            shutil.rmtree(cache_dir)
            print_success("Triton cache removed.")
        except Exception as e:
            print_warning(f"Could not remove Triton cache: {e}")
    else:
        print("No Triton cache found.")


def install_uv():
    """Install uv package manager."""
    print_header("Installing uv Package Manager")
    
    success, _ = run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        "Upgrading pip"
    )
    if not success:
        return False
    
    success, _ = run_command(
        [sys.executable, "-m", "pip", "install", "uv"],
        "Installing uv"
    )
    return success


def install_all_packages():
    """Install all packages with a single uv pip install command."""
    print_header("Installing All Packages")
    
    has_gpu = has_nvidia_gpu()
    
    if has_gpu:
        install_spec = "-e .[gpu]"
        print("Installing with GPU support")
    else:
        install_spec = "-e ."
        print("Installing CPU-only")
    
    success, output = run_command(
        ["uv", "pip", "install", install_spec],
        "Resolving and installing all dependencies",
        timeout=1800
    )
    
    if success:
        print_success("All packages installed successfully!")
    
    return success


def create_directory_structure():
    """Create required directory structure."""
    print_header("Creating Directory Structure")
    
    base_dir = Path(__file__).parent
    models_dir = base_dir / "Models"
    subdirs = ["chat", "vector", "vision", "whisper"]
    
    models_dir.mkdir(exist_ok=True)
    print(f"Created: {models_dir}")
    
    for subdir in subdirs:
        subdir_path = models_dir / subdir
        subdir_path.mkdir(exist_ok=True)
        print(f"Created: {subdir_path}")


def update_config_yaml():
    """Update config.yaml with default values."""
    print_header("Updating Configuration")
    
    try:
        import yaml
    except ImportError:
        print_warning("PyYAML not available, skipping config update.")
        return
    
    config_path = Path(__file__).parent / "config.yaml"
    
    # Load existing config or create empty
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    
    # Default configurations
    defaults = {
        "openai": {
            "api_key": "",
            "model": "gpt-4o-mini",
            "reasoning_effort": "medium",
        },
        "server": {
            "api_key": "",
            "connection_str": "http://localhost:1234/v1",
            "show_thinking": "medium",
        },
        "chatterbox": {
            "device": "auto",
        },
    }
    
    # Apply defaults (don't overwrite existing values)
    for section, values in defaults.items():
        if section not in config:
            config[section] = {}
        for key, default_value in values.items():
            if key not in config[section]:
                config[section][key] = default_value
    
    # Clean up server section (remove unauthorized keys)
    allowed_server_keys = {"api_key", "connection_str", "show_thinking"}
    config["server"] = {
        k: v for k, v in config["server"].items() 
        if k in allowed_server_keys
    }
    
    # Write config
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print_success(f"Configuration updated: {config_path}")


def run_post_install_hooks():
    """Run any post-installation hooks."""
    print_header("Running Post-Install Hooks")
    
    try:
        from replace_sourcecode import add_cuda_files
        add_cuda_files()
        print_success("CUDA files added.")
    except ImportError:
        print_warning("replace_sourcecode module not found, skipping.")
    except Exception as e:
        print_warning(f"Error in post-install hook: {e}")


def main():
    start_time = time.time()
    
    print_header("Windows Installation Script")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Step 1: Clean Triton cache
    clean_triton_cache()
    
    # Step 2: Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Step 3: Install uv
    if not install_uv():
        print_error("Failed to install uv. Exiting.")
        sys.exit(1)
    
    # Step 4: Install ALL packages in one command
    if not install_all_packages():
        print_error("Failed to install packages. Exiting.")
        sys.exit(1)
    
    # Step 5: Create directories
    create_directory_structure()
    
    # Step 6: Update config
    update_config_yaml()
    
    # Step 7: Post-install hooks
    run_post_install_hooks()
    
    # Step 8: Final Triton cache cleanup
    clean_triton_cache()
    
    # Summary
    print_header("Installation Complete")
    
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print_success(f"Total installation time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
    
    show_message_box(
        "Installation Complete",
        f"Installation finished in {int(minutes)} minutes {int(seconds)} seconds."
    )


if __name__ == "__main__":
    main()