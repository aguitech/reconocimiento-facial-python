"""Utilidades comunes: logging, validación de imágenes, conversiones BGR↔RGB."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image, UnidentifiedImageError


def configure_logging(level: str = "default") -> None:
    """Configura loguru con un formato compacto y sink a stderr.

    Args:
        level: Nivel de logging (TRACE, DEBUG, INFO, WARNING, ERROR).
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <7}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )


def load_image(image_path: str | Path, max_size: int | None = 800) -> np.ndarray:
    """Carga una imagen desde disco y la devuelve como arreglo RGB (uint8).

    Args:
        image_path: Ruta al archivo.
        max_size: Si se especifica, redimensiona manteniendo el aspecto de modo que el lado
                  más largo no supere `max_size`. Acelera el procesamiento en imágenes grandes.

    Returns:
        Arreglo NumPy con forma (H, W, 3) en formato RGB.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo no es una imagen válida.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Imagen no encontrada: {path}")

    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if max_size is not None and max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            return np.array(img, dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise ValueError(f"Archivo no es una imagen válida: {path}") from exc


def encode_image_b64(image: np.ndarray, format: str = "PNG") -> str:  # noqa: A002
    """Codifica un arreglo NumPy (RGB) a una cadena base64 (sin encabezado data:).

    Útil para enviar imágenes por JSON.
    """
    import base64
    from io import BytesIO

    pil_img = Image.fromarray(image)
    buffer = BytesIO()
    pil_img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image_b64(data: str) -> np.ndarray:
    """Decodifica una cadena base64 a un arreglo RGB.

    Acepta con o sin el prefijo `data:image/png;base64,...`.
    """
    import base64
    from io import BytesIO

    if "," in data and data.startswith("data:"):
        data = data.split(",", 1)[1]

    try:
        raw = base64.b64decode(data, validate=False)
        with Image.open(BytesIO(raw)) as img:
            return np.array(img.convert("RGB"), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"No se pudo decodificar la imagen base64: {exc}") from exc


def ensure_numpy_array(obj: Any, name: str = "input") -> np.ndarray:
    """Asegura que `obj` sea un ndarray NumPy; lanza error descriptivo si no."""
    if isinstance(obj, np.ndarray):
        return obj
    raise TypeError(
        f"{name} debe ser un ndarray NumPy, se recibió {type(obj).__name__}"
    )


__all__ = [
    "configure_logging",
    "load_image",
    "encode_image_b64",
    "decode_image_b64",
    "ensure_numpy_array",
    "logger",
]