import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from neurons.validator.utils.agent_storage import (
    MAX_AGENT_FILE_SIZE,
    get_agent_file_path,
    load_agent_code,
    save_agent_code,
    validate_hotkey,
    validate_miner_uid,
    verify_file_exists,
)


class TestValidateMinerUid:
    def test_valid_uid_zero(self):
        validate_miner_uid(0)

    def test_valid_uid_max(self):
        validate_miner_uid(256)

    def test_valid_uid_middle(self):
        validate_miner_uid(42)

    def test_invalid_uid_negative(self):
        with pytest.raises(ValueError, match="Must be 0-256"):
            validate_miner_uid(-1)

    def test_invalid_uid_too_high(self):
        with pytest.raises(ValueError, match="Must be 0-256"):
            validate_miner_uid(257)

    def test_invalid_uid_wrong_type(self):
        with pytest.raises(TypeError, match="must be int"):
            validate_miner_uid("42")


class TestValidateHotkey:
    def test_valid_hotkey(self):
        hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        validate_hotkey(hotkey)

    def test_invalid_hotkey_too_short(self):
        with pytest.raises(ValueError, match="Must be >= 40"):
            validate_hotkey("short")

    def test_invalid_hotkey_empty(self):
        with pytest.raises(ValueError, match="Must be >= 40"):
            validate_hotkey("")

    def test_invalid_hotkey_special_chars(self):
        with pytest.raises(ValueError, match="Must be >= 40"):
            validate_hotkey("5GrwvaEF/../../../etc/passwd")

    def test_invalid_hotkey_wrong_type(self):
        with pytest.raises(TypeError, match="must be str"):
            validate_hotkey(12345)


class TestGetAgentFilePath:
    def test_valid_path_construction(self):
        base = Path("/data/agents")
        uid = 42
        hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        version = uuid4()

        path = get_agent_file_path(base, uid, hotkey, version)

        assert path.is_absolute()
        assert str(path).startswith("/data/agents/42/")
        assert str(path).endswith(f"{version}.py")

    def test_path_is_relative_to_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            path = get_agent_file_path(
                base, 42, "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", uuid4()
            )
            assert path.is_relative_to(base)

    def test_invalid_base_dir_type(self):
        with pytest.raises(TypeError, match="base_dir must be Path"):
            get_agent_file_path("/data", 42, "hotkey", uuid4())

    def test_invalid_version_id_type(self):
        with pytest.raises(TypeError, match="version_id must be UUID"):
            get_agent_file_path(Path("/data"), 42, "hotkey", "not-a-uuid")

    def test_path_traversal_in_hotkey(self):
        with pytest.raises(ValueError, match="Must be >= 40"):
            get_agent_file_path(Path("/data"), 42, "../../../etc/passwd", uuid4())


class TestSaveAgentCode:
    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test" / "agent.py"
            code = b"def agent_main(): return 0.5"

            save_agent_code(file_path, code)

            assert file_path.exists()
            assert file_path.read_bytes() == code

    def test_save_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "a" / "b" / "c" / "agent.py"
            code = b"def agent_main(): pass"

            save_agent_code(file_path, code)

            assert file_path.exists()

    def test_save_sets_correct_permissions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "agent.py"
            code = b"def agent_main(): pass"

            save_agent_code(file_path, code)

            mode = file_path.stat().st_mode & 0o777
            assert mode == 0o644

    def test_invalid_file_path_type(self):
        with pytest.raises(TypeError, match="file_path must be Path"):
            save_agent_code("/tmp/agent.py", b"code")

    def test_invalid_code_bytes_type(self):
        with pytest.raises(TypeError, match="code_bytes must be bytes"):
            save_agent_code(Path("/tmp/agent.py"), "code")

    def test_empty_code_bytes(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            save_agent_code(Path("/tmp/agent.py"), b"")

    def test_code_exceeds_max_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "agent.py"
            large_code = b"x" * (MAX_AGENT_FILE_SIZE + 1)

            with pytest.raises(ValueError, match="exceeds maximum"):
                save_agent_code(file_path, large_code)


class TestVerifyFileExists:
    def test_file_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "agent.py"
            file_path.write_text("code")

            assert verify_file_exists(file_path) is True

    def test_file_does_not_exist(self):
        assert verify_file_exists(Path("/nonexistent/file.py")) is False

    def test_directory_not_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            assert verify_file_exists(dir_path) is False

    def test_invalid_path_type(self):
        assert verify_file_exists("/tmp/agent.py") is False


class TestLoadAgentCode:
    def test_load_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "agent.py"
            expected_code = "def agent_main(): return 0.5"
            file_path.write_text(expected_code, encoding="utf-8")

            code = load_agent_code(file_path)

            assert code == expected_code

    def test_invalid_file_path_type(self):
        with pytest.raises(TypeError, match="file_path must be Path"):
            load_agent_code("/tmp/agent.py")

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Agent file not found"):
            load_agent_code(Path("/nonexistent/agent.py"))

    def test_file_exceeds_max_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "agent.py"
            large_code = "x" * (MAX_AGENT_FILE_SIZE + 1)
            file_path.write_text(large_code, encoding="utf-8")

            with pytest.raises(ValueError, match="too large"):
                load_agent_code(file_path)

    def test_invalid_utf8_encoding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "agent.py"
            file_path.write_bytes(b"\x80\x81\x82")

            with pytest.raises(IOError, match="Invalid UTF-8 encoding"):
                load_agent_code(file_path)
