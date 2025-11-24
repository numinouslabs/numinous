from pathlib import Path

import docker
import docker.errors

from neurons.validator.utils.logger.logger import NuminousLogger


def build_docker_image(
    docker_client: docker.DockerClient,
    path: Path,
    image_tag: str,
    logger: NuminousLogger,
    quiet: bool = False,
) -> None:
    # Validate docker_client
    if not isinstance(docker_client, docker.DockerClient):
        raise TypeError("docker_client must be a DockerClient instance.")

    # Validate path
    if not isinstance(path, Path):
        raise TypeError("path must be a Path instance.")

    # Validate image_tag
    if not isinstance(image_tag, str) or not image_tag:
        raise ValueError("image_tag must be a non-empty string.")

    # Validate logger
    if not isinstance(logger, NuminousLogger):
        raise TypeError("logger must be an instance of NuminousLogger.")

    logger.info("Building Docker image", extra={"image_tag": image_tag, "path": str(path)})

    try:
        # Build Docker image using SDK
        _, build_logs = docker_client.images.build(
            path=str(path), tag=image_tag, rm=True, forcerm=True, quiet=quiet
        )

        # Log build output if not quiet
        if not quiet:
            for log in build_logs:
                if "stream" in log:
                    logger.debug("Docker build", extra={"output": log["stream"].strip()})
                elif "error" in log:
                    logger.error("Docker build error", extra={"error": log["error"]})

        logger.info("Docker image built successfully", extra={"image_tag": image_tag})

    except docker.errors.BuildError as e:
        logger.error(
            "Docker build failed",
            extra={"image_tag": image_tag, "error": str(e), "build_log": e.build_log},
        )
        raise RuntimeError(f"Docker build failed: {e}") from e
    except Exception as e:
        logger.error(
            "Failed to build Docker image", extra={"image_tag": image_tag, "error": str(e)}
        )
        raise


def image_exists(docker_client: docker.DockerClient, image_tag: str) -> bool:
    # Validate docker_client
    if not isinstance(docker_client, docker.DockerClient):
        raise TypeError("docker_client must be a DockerClient instance.")

    # Validate image_tag
    if not isinstance(image_tag, str) or not image_tag:
        raise ValueError("image_tag must be a non-empty string.")

    try:
        docker_client.images.get(image_tag)
        return True
    except docker.errors.ImageNotFound:
        return False
    except Exception:
        return False


def remove_image(
    docker_client: docker.DockerClient,
    image_tag: str,
    logger: NuminousLogger,
    force: bool = False,
) -> None:
    # Validate docker_client
    if not isinstance(docker_client, docker.DockerClient):
        raise TypeError("docker_client must be a DockerClient instance.")

    # Validate image_tag
    if not isinstance(image_tag, str) or not image_tag:
        raise ValueError("image_tag must be a non-empty string.")

    # Validate logger
    if not isinstance(logger, NuminousLogger):
        raise TypeError("logger must be an instance of NuminousLogger.")

    try:
        docker_client.images.remove(image=image_tag, force=force)
        logger.debug("Docker image removed", extra={"image_tag": image_tag})
    except docker.errors.ImageNotFound:
        logger.warning("Docker image not found", extra={"image_tag": image_tag})
    except Exception as e:
        logger.warning(
            "Error removing Docker image", extra={"image_tag": image_tag, "error": str(e)}
        )


def prune_images(docker_client: docker.DockerClient, logger: NuminousLogger) -> None:
    # Validate docker_client
    if not isinstance(docker_client, docker.DockerClient):
        raise TypeError("docker_client must be a DockerClient instance.")

    # Validate logger
    if not isinstance(logger, NuminousLogger):
        raise TypeError("logger must be an instance of NuminousLogger.")

    try:
        logger.info("Pruning unused Docker images")
        result = docker_client.images.prune()
        logger.debug(
            "Successfully pruned unused Docker images",
            extra={"space_reclaimed": result.get("SpaceReclaimed", 0)},
        )
    except Exception as e:
        logger.warning("Error pruning Docker images", extra={"error": str(e)})
