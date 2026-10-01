"""Pytest fixtures compartidos."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirige `RF_DATA_DIR` y `RF_DB_PATH` a un directorio temporal en cada test.

    Se aplica automáticamente para evitar que los tests escriban en el directorio real.
    """
    monkeypatch.setenv("RF_DATA_DIR", str(tmp_path / "rf_data"))
    monkeypatch.setenv("RF_DB_PATH", "test.db")
    # Quitar cualquier variable de log que pueda haberse colado.
    monkeypatch.delenv("RF_LOG_LEVEL", raising=False)