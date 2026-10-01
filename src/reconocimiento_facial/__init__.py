"""Sistema de reconocimiento facial en Python.

Módulos principales:
    - detector:    Detección de rostros (face_recognition / OpenCV).
    - encoder:     Codificación de rostros a embeddings 128-d.
    - database:    Almacenamiento de personas conocidas (SQLite).
    - recognizer:  Comparación de embeddings y clasificación.
    - api:         API REST con FastAPI.
    - cli:         Interfaz de línea de comandos con Typer.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Héctor Aguilar <hector@aguitech.com>"

__all__ = ["__version__", "__author__"]