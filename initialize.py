import platform
import shutil
from pathlib import Path
import logging

import torch
import yaml

from config import get_config

def get_compute_device_info():
    available_devices = ["cpu"]
    gpu_brand = None
    if torch.cuda.is_available():
        available_devices.append('cuda')

    return {
        'available': available_devices,
        'gpu_brand': gpu_brand
    }

def get_platform_info():
    return {'os': platform.system().lower()}

def update_config_file(**system_info):
    config = get_config()

    compute_device_info = system_info.get('Compute_Device', {})
    config.Compute_Device.available = compute_device_info.get('available', ['cpu'])

    for key in ['database_creation', 'database_query']:
        current = getattr(config.Compute_Device, key, 'cpu')
        if current not in ['cpu', 'cuda', 'mps']:
            setattr(config.Compute_Device, key, 'cpu')

    platform_info = system_info.get('Platform_Info', {})
    config.Platform_Info.os = platform_info.get('os', '')

    config.save()

def check_for_necessary_folders():
    config = get_config()
    folders = [
        config.root_dir / "Assets",
        config.docs_dir,
        config.vector_db_backup_dir,
        config.vector_db_dir,
        config.models_dir,
        config.vector_models_dir,
    ]

    for folder in folders:
        Path(folder).mkdir(exist_ok=True)

def restore_vector_db_backup():
    config = get_config()
    backup_folder = config.vector_db_backup_dir
    destination_folder = config.vector_db_dir

    if not backup_folder.exists():
        logging.error("Backup folder 'Vector_DB_Backup' does not exist.")
        return

    try:
        if destination_folder.exists():
            shutil.rmtree(destination_folder)
            logging.info("Deleted existing 'Vector_DB' folder.")
        destination_folder.mkdir()
        logging.info("Created 'Vector_DB' folder.")

        for item in backup_folder.iterdir():
            dest_path = destination_folder / item.name
            if item.is_dir():
                shutil.copytree(item, dest_path)
                logging.info(f"Copied directory: {item.name}")
            else:
                shutil.copy2(item, dest_path)
                logging.info(f"Copied file: {item.name}")
        logging.info("Successfully restored Vector DB backup.")
    except Exception as e:
        logging.error(f"Error restoring Vector DB backup: {e}")


def main():
    compute_device_info = get_compute_device_info()
    platform_info = get_platform_info()
    update_config_file(Compute_Device=compute_device_info, Platform_Info=platform_info)
    check_for_necessary_folders()

if __name__ == "__main__":
    main()
