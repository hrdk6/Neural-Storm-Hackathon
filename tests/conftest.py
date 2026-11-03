from __future__ import annotations

import logging
from pathlib import Path

import pytest

from clinictune.pipeline import build_demo


@pytest.fixture(scope="session")
def demo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A small synthetic demo bundle + onboarding packs, built once per test session."""
    logging.getLogger("clinictune").setLevel(logging.WARNING)
    root = tmp_path_factory.mktemp("demo")
    paths = build_demo(root, n_scale=0.2, seed=1)
    return {"root": root, **paths}
