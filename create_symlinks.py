from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Union, List, Tuple
import os
import shutil

def _create_single_symlink(args) -> Tuple[bool, str | None]:
    source_path, target_dir = args
    link_path = Path(target_dir) / Path(source_path).name
    try:
        if link_path.exists():
            return False, None

        try:
            link_path.symlink_to(source_path)
            return True, None
        except Exception as symlink_err:
            if os.name == "nt":
                try:
                    import ctypes
                    CreateHardLinkW = ctypes.windll.kernel32.CreateHardLinkW
                    CreateHardLinkW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_void_p]
                    CreateHardLinkW.restype = ctypes.c_bool
                    if CreateHardLinkW(str(link_path), str(source_path), None):
                        return True, None
                except Exception:
                    pass
            try:
                shutil.copy2(source_path, link_path)
                return True, None
            except Exception as copy_err:
                return False, (
                    f"Error creating link for {Path(source_path).name}: "
                    f"symlink failed ({symlink_err}); copy failed ({copy_err})"
                )
    except Exception as e:
        return False, f"Error creating link for {Path(source_path).name}: {e}"

def create_symlinks_parallel(source: Union[str, Path, List[str], List[Path]], 
                           target_dir: Union[str, Path] = "Docs_for_DB") -> Tuple[int, list]:
    target_dir = Path(target_dir)
    if not target_dir.exists():
        print(f"Target directory does not exist: {target_dir}")
        return 0, []

    try:
        if isinstance(source, (str, Path)) and not isinstance(source, list):
            source_dir = Path(source)
            if not source_dir.exists():
                raise ValueError(f"Source directory does not exist: {source_dir}")
            files = [(str(p), str(target_dir)) for p in source_dir.iterdir() if p.is_file()]

        elif isinstance(source, list):
            files = [(str(Path(p)), str(target_dir)) for p in source]

        else:
            raise ValueError("Source must be either a directory path or a list of file paths")

        file_count = len(files)
        if file_count <= 500:
            results = [_create_single_symlink(file) for file in files]
        else:

            if file_count <= 5000:
                processes = 2
            elif file_count <= 20000:
                processes = 3
            else:
                processes = 4

            print(f"Processing {file_count} files using {processes} processes")

            with Pool(processes=processes) as pool:
                results = pool.map(_create_single_symlink, files)

        count = sum(1 for success, _ in results if success)
        errors = [error for _, error in results if error is not None]

        print(f"\nComplete! Created {count} symbolic links")
        if errors:
            print("\nErrors occurred:")
            for error in errors:
                print(error)

        return count, errors

    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")