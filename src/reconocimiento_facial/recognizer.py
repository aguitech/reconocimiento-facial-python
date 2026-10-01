"""Reconocedor: compara embeddings y los asigna a personas registradas.

Dos métricas de distancia:
    - `euclidean`:     distancia L1/L2 cruda (no normalizada).
    - `euclidean_l2`:  distancia euclídea entre vectores normalizados (recomendado por dlib).

Un rostro se considera reconocido si la menor distancia es menor a la tolerancia.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .database import FaceDatabase, Person
from .encoder import FaceEncoding
from .utils import logger


@dataclass(frozen=True, slots=True)
class Match:
    """Resultado de comparar un rostro contra todas las muestras registradas."""

    person: Person | None
    distance: float
    confidence: float
    encoding: FaceEncoding
    candidates: list[tuple[Person, float]] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        """True si el rostro fue identificado (distancia menor al umbral)."""
        return self.person is not None


class FaceRecognizer:
    """Compara embeddings contra la base de datos."""

    def __init__(
        self,
        database: FaceDatabase,
        tolerance: float = 0.6,
        metric: str = "euclidean_l2",
    ) -> None:
        """Inicializa el reconocedor.

        Args:
            database: Base de datos de personas.
            tolerance: Umbral de distancia. Menor = más estricto.
                       - 0.6 es el valor por defecto recomendado por face_recognition.
                       - 0.5 reduce falsos positivos.
                       - 0.4 muy estricto (puede fallar con fotos reales).
            metric: "euclidean" o "euclidean_l2".
        """
        if metric not in {"euclidean", "euclidean_l2"}:
            raise ValueError(f"Métrica no soportada: {metric!r}")
        if not (0.0 <= tolerance <= 1.0):
            raise ValueError(f"Tolerancia fuera de rango: {tolerance}")

        self.database = database
        self.tolerance = tolerance
        self.metric = metric
        self._known_persons: list[Person] = []
        self._known_encodings: list[FaceEncoding] = []
        self._dirty = True

        logger.debug(
            "FaceRecognizer inicializado (tolerance={}, metric={})",
            tolerance,
            metric,
        )

    # --- Carga / recarga del cache de personas conocidas -------------------

    def reload(self) -> int:
        """Recarga los embeddings conocidos desde la base de datos.

        Returns:
            Número de muestras cargadas.
        """
        persons: list[Person] = []
        encodings: list[FaceEncoding] = []

        for person, samples in self.database.iter_all_for_training():
            persons.extend([person] * len(samples))
            encodings.extend(samples)

        self._known_persons = persons
        self._known_encodings = encodings
        self._dirty = False
        logger.info("Cache de reconocimiento cargado: {} muestras", len(encodings))
        return len(encodings)

    def mark_dirty(self) -> None:
        """Marca el cache como desactualizado (llamar tras añadir/eliminar muestras)."""
        self._dirty = True

    @property
    def n_known(self) -> int:
        """Número de muestras conocidas cargadas."""
        if self._dirty:
            self.reload()
        return len(self._known_encodings)

    # --- Distancias ----------------------------------------------------------

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
        """Calcula la distancia entre dos embeddings."""
        if metric == "euclidean":
            return float(np.linalg.norm(a - b))
        if metric == "euclidean_l2":
            norm_a = a / (np.linalg.norm(a) + 1e-10)
            norm_b = b / (np.linalg.norm(b) + 1e-10)
            return float(np.linalg.norm(norm_a - norm_b))
        raise ValueError(f"Métrica no soportada: {metric!r}")

    # --- Comparación ---------------------------------------------------------

    def _best_match(self, encoding: FaceEncoding) -> tuple[Person | None, float, list]:
        """Encuentra la mejor coincidencia para un encoding.

        Returns:
            (persona, distancia, [(p, dist), ...] ordenadas por distancia)
        """
        if self._dirty:
            self.reload()

        if not self._known_encodings:
            return None, float("inf"), []

        query = encoding.vector
        distances: list[tuple[Person, float]] = []
        for person, known in zip(self._known_persons, self._known_encodings, strict=False):
            d = self._distance(query, known.vector, self.metric)
            distances.append((person, d))

        distances.sort(key=lambda x: x[1])
        best_person, best_distance = distances[0]
        return best_person, best_distance, distances

    def recognize(self, encoding: FaceEncoding) -> Match:
        """Reconoce un embedding contra la base.

        Args:
            encoding: Encoding a clasificar.

        Returns:
            `Match` con la mejor coincidencia, distancia, confianza y ranking de candidatos.
        """
        best_person, distance, candidates = self._best_match(encoding)
        confidence = self._distance_to_confidence(distance)

        matched = best_person is not None and distance <= self.tolerance
        return Match(
            person=best_person if matched else None,
            distance=distance if matched else float("nan"),
            confidence=confidence,
            encoding=encoding,
            candidates=candidates[:5],
        )

    def recognize_batch(self, encodings: Iterable[FaceEncoding]) -> list[Match]:
        """Reconoce una lista de encodings."""
        return [self.recognize(e) for e in encodings]

    @staticmethod
    def _distance_to_confidence(distance: float) -> float:
        """Convierte una distancia [0, 1] a confianza [0, 1] (1 = muy seguro).

        Es una función monotónica decreciente aproximada:
            - distancia 0.0  → confianza 1.0
            - distancia 0.4  → confianza 0.6
            - distancia 0.6  → confianza 0.0
            - distancia >0.6 → confianza negativa (sin coincidencia)
        """
        if np.isinf(distance):
            return 0.0
        # Mapeo lineal con clipping.
        conf = 1.0 - (distance / 0.6)
        return float(max(0.0, conf))


__all__ = ["FaceRecognizer", "Match"]