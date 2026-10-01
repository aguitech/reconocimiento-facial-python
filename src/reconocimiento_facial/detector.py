"""Detector de rostros.

Soporta dos backends:
    - `face_recognition`: usa dlib HOG/SVM. Más preciso, requiere compilación.
    - `opencv_haar`:      Haar cascades de OpenCV. Más rápido, menos preciso.

Ambos devuelven una lista de `FaceLocation` con bounding boxes en formato
`(top, right, bottom, left)` igual al de `face_recognition.face_locations`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .utils import logger


@dataclass(frozen=True, slots=True)
class FaceLocation:
    """Localización de un rostro detectado en una imagen.

    Attributes:
        top, left, bottom, right: Coordenadas del rectángulo en píxeles.
        confidence: Confianza estimada (0.0–1.0). Solo disponible para algunos backends.
    """

    top: int
    right: int
    bottom: int
    left: int
    confidence: float | None = None

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Convierte a la tupla (top, right, bottom, left) usada por face_recognition."""
        return (self.top, self.right, self.bottom, self.left)

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


class FaceDetector(Protocol):
    """Protocolo para implementaciones de detector."""

    def detect(self, image: np.ndarray) -> list[FaceLocation]:
        """Detecta rostros en `image` (RGB, uint8)."""
        ...


class FaceRecognitionDetector:
    """Detector basado en `face_recognition` (dlib)."""

    def __init__(self, model: str = "hog") -> None:
        """Inicializa el detector.

        Args:
            model: Modelo de detección. "hog" es rápido y CPU-only, "cnn" requiere GPU.
        """
        if model not in {"hog", "cnn"}:
            raise ValueError(f"Modelo no soportado: {model!r}")
        self.model = model
        # Importación perezosa para que `opencv_haar` no requiera dlib.
        import face_recognition  # noqa: F401

        logger.debug("FaceRecognitionDetector inicializado (model={})", model)

    def detect(self, image: np.ndarray) -> list[FaceLocation]:
        """Detecta rostros usando face_recognition.

        Args:
            image: Arreglo RGB (H, W, 3), uint8.

        Returns:
            Lista de `FaceLocation`.
        """
        import face_recognition

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Imagen debe ser (H, W, 3), se recibió {image.shape}")

        locations = face_recognition.face_locations(image, model=self.model)
        return [FaceLocation(top=t, right=r, bottom=b, left=l) for (t, r, b, l) in locations]


class OpenCVHaarDetector:
    """Detector basado en Haar cascades de OpenCV.

    Más rápido que `face_recognition` pero menos preciso y sensible a oclusiones.
    Útil como fallback si dlib no se puede compilar.
    """

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5) -> None:
        """Carga el clasificador Haar preentrenado de OpenCV.

        Raises:
            RuntimeError: Si el archivo XML no se encuentra (instalación corrupta).
        """
        import cv2

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            raise RuntimeError(f"Haar cascade no encontrado: {cascade_path}")

        self.classifier = cv2.CascadeClassifier(str(cascade_path))
        if self.classifier.empty():
            raise RuntimeError(f"No se pudo cargar el Haar cascade: {cascade_path}")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        logger.debug("OpenCVHaarDetector inicializado (cascade={})", cascade_path.name)

    def detect(self, image: np.ndarray) -> list[FaceLocation]:
        """Detecta rostros en una imagen RGB.

        OpenCV espera BGR o grayscale; convertimos internamente.
        """
        import cv2

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Imagen debe ser (H, W, 3), se recibió {image.shape}")

        # Convertir a grayscale para Haar.
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        rects = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        locations: list[FaceLocation] = []
        for (x, y, w, h) in rects:
            # Convertir a (top, right, bottom, left)
            locations.append(FaceLocation(top=int(y), right=int(x + w), bottom=int(y + h), left=int(x)))

        return locations


def build_detector(backend: str = "face_recognition", **kwargs) -> FaceDetector:
    """Fábrica de detectores.

    Args:
        backend: "face_recognition" u "opencv_haar".
        **kwargs: Argumentos específicos del backend.

    Returns:
        Instancia de `FaceDetector`.
    """
    if backend == "face_recognition":
        return FaceRecognitionDetector(**kwargs)
    if backend == "opencv_haar":
        return OpenCVHaarDetector(**kwargs)
    raise ValueError(f"Backend no soportado: {backend!r}. Usa 'face_recognition' u 'opencv_haar'.")


__all__ = [
    "FaceDetector",
    "FaceLocation",
    "FaceRecognitionDetector",
    "OpenCVHaarDetector",
    "build_detector",
]