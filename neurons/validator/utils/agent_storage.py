from pathlib import Path
from typing import Final
from uuid import UUID

MAX_AGENT_FILE_SIZE: Final[int] = 2 * 1024 * 1024


def validate_miner_uid(miner_uid: int) -> None:
    if not isinstance(miner_uid, int):
        raise TypeError(f"miner_uid must be int, got {type(miner_uid).__name__}")

    if not (0 <= miner_uid <= 256):
        raise ValueError(f"Invalid miner_uid: {miner_uid}. Must be 0-256.")


def validate_hotkey(hotkey: str) -> None:
    """Validate hotkey format to prevent path traversal attacks."""
    if not isinstance(hotkey, str):
        raise TypeError(f"hotkey must be str, got {type(hotkey).__name__}")

    if not hotkey or len(hotkey) < 40:
        raise ValueError(
            f"Invalid hotkey length: {len(hotkey) if hotkey else 0}. " f"Must be >= 40 characters."
        )

    # SS58 addresses are alphanumeric only
    if not hotkey.isalnum():
        raise ValueError(f"Hotkey must be alphanumeric (base58): {hotkey}")


def get_agent_file_path(
    base_dir: Path,
    miner_uid: int,
    hotkey: str,
    version_id: UUID,
) -> Path:
    if not isinstance(base_dir, Path):
        raise TypeError(f"base_dir must be Path, got {type(base_dir).__name__}")

    if not isinstance(version_id, UUID):
        raise TypeError(f"version_id must be UUID, got {type(version_id).__name__}")

    validate_miner_uid(miner_uid)
    validate_hotkey(hotkey)

    path = base_dir / str(miner_uid) / hotkey / f"{version_id}.py"

    try:
        resolved = path.resolve()
        base_resolved = base_dir.resolve()

        if not resolved.is_relative_to(base_resolved):
            raise ValueError(f"Path traversal detected: {path} is not relative to {base_dir}")
    except OSError as e:
        raise ValueError(f"Invalid path construction: {e}")

    return path


def save_agent_code(
    file_path: Path, code_bytes: bytes, max_size: int = MAX_AGENT_FILE_SIZE
) -> None:
    if not isinstance(file_path, Path):
        raise TypeError(f"file_path must be Path, got {type(file_path).__name__}")

    if not isinstance(code_bytes, bytes):
        raise TypeError(f"code_bytes must be bytes, got {type(code_bytes).__name__}")

    if not isinstance(max_size, int):
        raise TypeError(f"max_size must be int, got {type(max_size).__name__}")

    if len(code_bytes) == 0:
        raise ValueError("code_bytes cannot be empty")

    if len(code_bytes) > max_size:
        raise ValueError(
            f"Agent code size {len(code_bytes)} bytes exceeds maximum {max_size} bytes"
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_bytes(code_bytes)

    file_path.chmod(0o644)

    if not file_path.exists():
        raise IOError(f"Failed to write file: {file_path}")

    actual_size = file_path.stat().st_size
    if actual_size != len(code_bytes):
        raise IOError(
            f"File size mismatch for {file_path}: "
            f"expected {len(code_bytes)} bytes, got {actual_size} bytes"
        )


def verify_file_exists(file_path: Path) -> bool:
    if not isinstance(file_path, Path):
        return False

    return file_path.exists() and file_path.is_file()


def load_agent_code(file_path: Path, max_size: int = MAX_AGENT_FILE_SIZE) -> str:
    if not isinstance(file_path, Path):
        raise TypeError(f"file_path must be Path, got {type(file_path).__name__}")

    if not file_path.exists():
        raise FileNotFoundError(f"Agent file not found: {file_path}")

    file_size = file_path.stat().st_size
    if file_size > max_size:
        raise ValueError(f"Agent file too large: {file_size} bytes (max: {max_size} bytes)")

    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise IOError(f"Invalid UTF-8 encoding: {e}")
    except Exception as e:
        raise IOError(f"Failed to read agent file: {e}")
