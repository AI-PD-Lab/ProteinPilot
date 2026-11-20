"""Shared pytest configuration hooks for foundry tests."""

import rootutils
from dotenv import load_dotenv


def configure_pytest(config, conftest_file: str) -> None:
    """Configure pytest for foundry tests.
    Sets up project root and environment variables.
    """
    rootutils.setup_root(conftest_file, indicator=".project-root", pythonpath=True)

    load_dotenv(override=True)
