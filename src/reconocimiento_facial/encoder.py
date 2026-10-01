"""Codificación de rostros a embeddings 128-d usando dlib.

Cada embedding es un vector NumPy de 128 floats. La distancia entre dos embeddings
indica qué tan parecidos son dos rostros: menor distancia = más parecidos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .detector import FaceLocation
from .utils import logger


@dataclass(frozen=True, slots=True)
class FaceEncoding:
    """Embedding facial inmutable con metadatos opcionales.

    Attributes:
        vector: Vector NumPy de 128 floats.
        location: Ubicación original del rostro (opcional).
        person_id: ID de la persona asociada, si fue etiquetado.
    """

    vector: np.ndarray
    location: FaceLocation | None = None
    person_id: str | None = None

    def __post_init__(self) -> None:
        # `frozen=True` impide setattr, pero podemos validar antes.
        if not isinstance(self.vector, np.ndarray):
            raise TypeError(f"vector debe ser ndarray, se recibió {type(self.vector).__name__}")
        if self.vector.shape != (128,):
            raise ValueError(f"vector debe tener forma (128,), se recibió {self.vector.shape}")
        if self.vector.dtype != np.float64:
            # Convertir silenciosamente para mantener el contrato.
            object.__setattr__(self, "vector", self.vector.astype(np.float64))

    def to_list(self) -> list[float]:
        """Convierte el vector a lista de Python (serializable JSON)."""
        return self.vector.tolist()

    @classmethod
    def from_list(cls, values: list[float], **kwargs) -> "FaceEncoding":
        """Reconstruye un encoding desde una lista (ej. desde JSON)."""
        return cls(vector=np.array(values, dtype=np.float64), **kwargs)


class FaceEncoder:
    """Calcula embeddings 128-dimensionales de rostros."""

    def __init__(self, model: str = "small") -> None:
        """Inicializa el encoder.

        Args:
            model: "small" (5 landmarks) o "large" (68 landmarks). Más grande = más preciso
                   y más lento.
        """
        if model not in {"small", "large"}:
            raise ValueError(f"Modelo de encoder no soportado: {model!r}")
        self.model = model
        # Importación perezosa por la misma razón que en el detector.
        import face_recognition  # noqa: F401

        logger.debug("FaceEncoder inicializado (model={})", model)

    def encode(
        self,
        image: np.ndarray,
        locations: Iterable[FaceLocation] | None = None,
        num_jitters: int = 1,
    ) -> list[FaceEncoding]:
        """Calcula embeddings de uno o más rostros.

        Args:
            image: Arreglo RGB (H, W, 3), uint8.
            locations: Bounding boxes. Si es None, detecta automáticamente.
            num_jitters: Veces que se muestrea cada rostro. 1 = rápido, 100 = muy preciso.

        Returns:
            Lista de `FaceEncoding` en el mismo orden que las detecciones.
        """
        import face_recognition

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Imagen debe ser (H, W, 3), se recibió {image.shape}")

        if locations is None:
            locations = face_recognition.face_locations(image, model="hog")
            locations = [FaceLocation(top=t, right=r, bottom=b, left=l) for (t, r, b, l) in locations]

        locations_list = list(locations)
        if not locations_list:
            return []

        # `face_recognition.face_encodings` acepta la lista de tuplas.
        tuples = [loc.to_tuple() for loc in locations_list]
        vectors = face_recognition.face_encodings(
            image,
            known_face_locations=tuples,
            num_jitters=num_jitters,
            model=self.model,
        )

        encodings: list[FaceEncoding] = []
        for loc, vec in zip(locations_list, vectors, strict=True):
            encodings.append(FaceEncoding(vector=vec, location=loc))

        return encodings


__all__ = ["FaceEncoder", "FaceEncoding"]