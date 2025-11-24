import os
import shutil
import stat
import tempfile
from pathlib import Path


def create_temp_dir(prefix: str = "ig_validator_sandbox_", base_dir: Path = None) -> Path:
    # Validate prefix
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string.")

    # Validate base_dir
    if base_dir is not None:
        if not isinstance(base_dir, Path):
            raise TypeError("base_dir must be a Path instance or None.")
        # Ensure base directory exists
        base_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix=prefix, dir=str(base_dir))
    else:
        temp_dir = tempfile.mkdtemp(prefix=prefix)

    return Path(temp_dir)


def cleanup_temp_dir(temp_path: Path) -> None:
    # Validate temp_path
    if not isinstance(temp_path, Path):
        raise TypeError("temp_path must be a Path instance.")

    if not temp_path.exists():
        return

    try:

        def make_writable(path: Path) -> None:
            try:
                os.chmod(path, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
            except Exception:
                pass

        # Walk through all files and make them writable
        for root, dirs, files in os.walk(temp_path):
            root_path = Path(root)
            make_writable(root_path)

            for d in dirs:
                make_writable(root_path / d)

            for f in files:
                make_writable(root_path / f)

        # Remove the directory
        shutil.rmtree(temp_path, ignore_errors=True)

    except Exception:
        # If cleanup fails, just ignore - temp directories will be cleaned up by OS eventually
        pass


def get_temp_dir_size(temp_path: Path) -> int:
    # Validate temp_path
    if not isinstance(temp_path, Path):
        raise TypeError("temp_path must be a Path instance.")

    total_size = 0
    try:
        for dirpath, _, filenames in os.walk(temp_path):
            for filename in filenames:
                filepath = Path(dirpath) / filename
                if filepath.exists():
                    total_size += filepath.stat().st_size
    except Exception:
        pass

    return total_size
