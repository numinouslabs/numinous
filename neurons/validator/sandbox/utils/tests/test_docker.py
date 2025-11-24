from pathlib import Path
from unittest.mock import MagicMock

import docker
import docker.errors
import pytest

from neurons.validator.sandbox.utils.docker import (
    build_docker_image,
    image_exists,
    prune_images,
    remove_image,
)
from neurons.validator.utils.logger.logger import NuminousLogger


@pytest.fixture
def mock_docker_client():
    return MagicMock(spec=docker.DockerClient)


@pytest.fixture
def mock_logger():
    logger = MagicMock(spec=NuminousLogger)
    return logger


class TestBuildDockerImage:
    def test_invalid_docker_client_type(self, mock_logger):
        with pytest.raises(TypeError, match="docker_client must be a DockerClient instance"):
            build_docker_image("not_a_client", Path("/tmp"), "test:latest", mock_logger)

    def test_invalid_path_type(self, mock_docker_client, mock_logger):
        with pytest.raises(TypeError, match="path must be a Path instance"):
            build_docker_image(mock_docker_client, "/tmp/path", "test:latest", mock_logger)

    def test_empty_image_tag(self, mock_docker_client, mock_logger):
        with pytest.raises(ValueError, match="image_tag must be a non-empty string"):
            build_docker_image(mock_docker_client, Path("/tmp"), "", mock_logger)

    def test_invalid_logger_type(self, mock_docker_client):
        with pytest.raises(TypeError, match="logger must be an instance of NuminousLogger"):
            build_docker_image(mock_docker_client, Path("/tmp"), "test:latest", "not_a_logger")

    def test_build_success_quiet_mode(self, mock_docker_client, mock_logger):
        mock_image = MagicMock()
        mock_docker_client.images.build.return_value = (mock_image, [])

        build_docker_image(mock_docker_client, Path("/tmp"), "test:latest", mock_logger, quiet=True)

        mock_docker_client.images.build.assert_called_once_with(
            path="/tmp", tag="test:latest", rm=True, forcerm=True, quiet=True
        )
        mock_logger.info.assert_called()

    def test_build_error_raises_runtime_error(self, mock_docker_client, mock_logger):
        mock_docker_client.images.build.side_effect = docker.errors.BuildError(
            "Build failed", build_log=[]
        )

        with pytest.raises(RuntimeError, match="Docker build failed"):
            build_docker_image(mock_docker_client, Path("/tmp"), "test:latest", mock_logger)

        mock_logger.error.assert_called()


class TestImageExists:
    def test_invalid_docker_client_type(self):
        with pytest.raises(TypeError, match="docker_client must be a DockerClient instance"):
            image_exists("not_a_client", "test:latest")

    def test_empty_image_tag(self, mock_docker_client):
        with pytest.raises(ValueError, match="image_tag must be a non-empty string"):
            image_exists(mock_docker_client, "")

    def test_existing_image_returns_true(self, mock_docker_client):
        mock_docker_client.images.get.return_value = MagicMock()

        assert image_exists(mock_docker_client, "test:latest") is True

    def test_missing_image_returns_false(self, mock_docker_client):
        mock_docker_client.images.get.side_effect = docker.errors.ImageNotFound("Not found")

        assert image_exists(mock_docker_client, "test:latest") is False

    def test_generic_error_returns_false(self, mock_docker_client):
        mock_docker_client.images.get.side_effect = Exception("API error")

        assert image_exists(mock_docker_client, "test:latest") is False


class TestRemoveImage:
    def test_invalid_docker_client_type(self, mock_logger):
        with pytest.raises(TypeError, match="docker_client must be a DockerClient instance"):
            remove_image("not_a_client", "test:latest", mock_logger)

    def test_empty_image_tag(self, mock_docker_client, mock_logger):
        with pytest.raises(ValueError, match="image_tag must be a non-empty string"):
            remove_image(mock_docker_client, "", mock_logger)

    def test_invalid_logger_type(self, mock_docker_client):
        with pytest.raises(TypeError, match="logger must be an instance of NuminousLogger"):
            remove_image(mock_docker_client, "test:latest", "not_a_logger")

    def test_remove_existing_image(self, mock_docker_client, mock_logger):
        remove_image(mock_docker_client, "test:latest", mock_logger)

        mock_docker_client.images.remove.assert_called_once_with(image="test:latest", force=False)
        mock_logger.debug.assert_called()

    def test_remove_with_force_flag(self, mock_docker_client, mock_logger):
        remove_image(mock_docker_client, "test:latest", mock_logger, force=True)

        mock_docker_client.images.remove.assert_called_once_with(image="test:latest", force=True)

    def test_image_not_found_logs_warning(self, mock_docker_client, mock_logger):
        mock_docker_client.images.remove.side_effect = docker.errors.ImageNotFound("Not found")

        remove_image(mock_docker_client, "test:latest", mock_logger)

        mock_logger.warning.assert_called()


class TestPruneImages:
    def test_invalid_docker_client_type(self, mock_logger):
        with pytest.raises(TypeError, match="docker_client must be a DockerClient instance"):
            prune_images("not_a_client", mock_logger)

    def test_invalid_logger_type(self, mock_docker_client):
        with pytest.raises(TypeError, match="logger must be an instance of NuminousLogger"):
            prune_images(mock_docker_client, "not_a_logger")

    def test_prune_success(self, mock_docker_client, mock_logger):
        mock_docker_client.images.prune.return_value = {"SpaceReclaimed": 1024}

        prune_images(mock_docker_client, mock_logger)

        mock_docker_client.images.prune.assert_called_once()
        mock_logger.info.assert_called()
        mock_logger.debug.assert_called()

    def test_prune_error_logs_warning(self, mock_docker_client, mock_logger):
        mock_docker_client.images.prune.side_effect = Exception("Prune failed")

        prune_images(mock_docker_client, mock_logger)

        mock_logger.warning.assert_called()
