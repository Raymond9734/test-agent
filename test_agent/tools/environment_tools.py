# test_agent/tools/environment_tools.py

import os
import shutil
import logging
import tempfile
import subprocess
from typing import List, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)


def setup_environment(
    language: str, project_dir: str, temp_dir: Optional[str] = None
) -> Tuple[bool, str, str]:
    """
    Set up a test environment for a specific language.

    Args:
        language: Programming language ("python", "go", etc.)
        project_dir: Project directory
        temp_dir: Optional temporary directory to use (if None, creates one)

    Returns:
        Tuple of (success flag, environment path, message)
    """
    if temp_dir is None:
        temp_dir = os.path.join(
            tempfile.gettempdir(),
            f"test_agent_{language}_{os.path.basename(project_dir)}",
        )

    # Create the temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)

    logger.info(f"Setting up {language} environment in {temp_dir}")

    if language.lower() == "python":
        return _setup_python_environment(project_dir, temp_dir)
    elif language.lower() == "go":
        return _setup_go_environment(project_dir, temp_dir)
    else:
        return False, temp_dir, f"Unsupported language: {language}"


def _setup_python_environment(project_dir: str, env_dir: str) -> Tuple[bool, str, str]:
    """
    Set up a Python virtual environment.

    Args:
        project_dir: Project directory
        env_dir: Environment directory

    Returns:
        Tuple of (success flag, environment path, message)
    """
    try:
        # Create virtual environment
        import venv

        venv.create(env_dir, with_pip=True, clear=True)

        # Get path to pip
        if os.name == "nt":  # Windows
            pip_path = os.path.join(env_dir, "Scripts", "pip")
            # python_path = os.path.join(env_dir, "Scripts", "python")
        else:  # Unix/Linux/Mac
            pip_path = os.path.join(env_dir, "bin", "pip")
            # python_path = os.path.join(env_dir, "bin", "python")

        # Install pytest
        logger.info(f"Installing pytest in {env_dir}")
        subprocess.run(
            [pip_path, "install", "-U", "pytest"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Check for requirements file and install if present
        requirements_file = os.path.join(project_dir, "requirements.txt")
        if os.path.exists(requirements_file):
            logger.info(f"Installing dependencies from {requirements_file}")
            try:
                subprocess.run(
                    [pip_path, "install", "-r", requirements_file],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Error installing project requirements: {e}")
                # Continue even if requirements installation fails
                return (
                    True,
                    env_dir,
                    f"Environment created but some dependencies failed: {e}",
                )

        # Check for setup.py or pyproject.toml
        setup_py = os.path.join(project_dir, "setup.py")
        pyproject_toml = os.path.join(project_dir, "pyproject.toml")

        if os.path.exists(setup_py):
            logger.info(f"Installing project in development mode from {setup_py}")
            try:
                subprocess.run(
                    [pip_path, "install", "-e", project_dir],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Error installing project in development mode: {e}")
        elif os.path.exists(pyproject_toml):
            logger.info(f"Installing project from {pyproject_toml}")
            try:
                subprocess.run(
                    [pip_path, "install", "-e", project_dir],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Error installing project from pyproject.toml: {e}")

        return True, env_dir, f"Python environment created at {env_dir}"

    except Exception as e:
        logger.error(f"Error setting up Python environment: {e}")
        return False, env_dir, f"Failed to create Python environment: {str(e)}"


def _setup_go_environment(project_dir: str, env_dir: str) -> Tuple[bool, str, str]:
    """
    Set up a Go environment.

    Args:
        project_dir: Project directory
        env_dir: Environment directory

    Returns:
        Tuple of (success flag, environment path, message)
    """
    try:
        # Check if Go is installed
        try:
            subprocess.run(
                ["go", "version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Go is not installed or not in PATH")
            return False, env_dir, "Go is not installed or not in PATH"

        # Create a temporary GOPATH
        go_path = os.path.join(env_dir, "gopath")
        os.makedirs(go_path, exist_ok=True)

        # Set environment variables
        env = os.environ.copy()
        env["GOPATH"] = go_path

        # Check if the project has go.mod
        go_mod = os.path.join(project_dir, "go.mod")
        if os.path.exists(go_mod):
            # Download dependencies
            logger.info(f"Downloading Go dependencies for {project_dir}")
            try:
                subprocess.run(
                    ["go", "mod", "download"],
                    check=True,
                    cwd=project_dir,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Error downloading Go dependencies: {e}")
                return (
                    True,
                    env_dir,
                    f"Go environment created but dependency download failed: {e}",
                )

        return True, env_dir, f"Go environment created at {env_dir}"

    except Exception as e:
        logger.error(f"Error setting up Go environment: {e}")
        return False, env_dir, f"Failed to create Go environment: {str(e)}"


def install_dependencies(
    environment_path: str, dependencies: List[str], language: str
) -> Tuple[bool, str]:
    """
    Install dependencies in the test environment.

    Args:
        environment_path: Path to the environment
        dependencies: List of dependencies to install
        language: Programming language

    Returns:
        Tuple of (success flag, message)
    """
    if not dependencies:
        return True, "No dependencies to install"

    logger.info(f"Installing {len(dependencies)} dependencies for {language}")

    try:
        if language.lower() == "python":
            # Get path to pip
            if os.name == "nt":  # Windows
                pip_path = os.path.join(environment_path, "Scripts", "pip")
            else:  # Unix/Linux/Mac
                pip_path = os.path.join(environment_path, "bin", "pip")

            # Install dependencies
            for dependency in dependencies:
                try:
                    subprocess.run(
                        [pip_path, "install", dependency],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to install {dependency}: {e}")
                    return False, f"Failed to install {dependency}: {e}"

            return (
                True,
                f"Successfully installed {len(dependencies)} Python dependencies",
            )

        elif language.lower() == "go":
            # Set environment variables
            env = os.environ.copy()
            env["GOPATH"] = os.path.join(environment_path, "gopath")

            # Install dependencies using go get
            for dependency in dependencies:
                try:
                    subprocess.run(
                        ["go", "get", dependency],
                        check=True,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to install {dependency}: {e}")
                    return False, f"Failed to install {dependency}: {e}"

            return True, f"Successfully installed {len(dependencies)} Go dependencies"

        else:
            return False, f"Dependency installation not implemented for {language}"

    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return False, f"Error installing dependencies: {str(e)}"


def cleanup_environment(environment_path: str) -> Tuple[bool, str]:
    """
    Clean up a test environment.

    Args:
        environment_path: Path to the environment

    Returns:
        Tuple of (success flag, message)
    """
    if not os.path.exists(environment_path):
        return True, f"Environment {environment_path} does not exist"

    try:
        logger.info(f"Cleaning up environment at {environment_path}")
        shutil.rmtree(environment_path, ignore_errors=True)
        return True, f"Successfully cleaned up environment at {environment_path}"
    except Exception as e:
        logger.error(f"Error cleaning up environment: {e}")
        return False, f"Error cleaning up environment: {str(e)}"
