"""Pipelines de alto nivel: reconocimiento en imágenes y webcam."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import get_settings
from .database import FaceDatabase, Person
from .detector import FaceDetector, FaceLocation, build_detector
from .encoder import FaceEncoder, FaceEncoding
from .recognizer import FaceRecognizer, Match
from .utils import logger


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    """Resultado de reconocimiento sobre una imagen completa."""

    image_path: str | None
    faces: list[Match]
    image_width: int
    image_height: int
    elapsed_ms: float


class FacePipeline:
    """Orquesta detección → encoding → reconocimiento."""

    def __init__(
        self,
        database: FaceDatabase | None = None,
        detector: FaceDetector | None = None,
        encoder: FaceEncoder | None = None,
        recognizer: FaceRecognizer | None = None,
    ) -> None:
        """Inicializa el pipeline.

        Si pasas componentes, se usan tal cual; si no, se construyen desde la config.
        """
        settings = get_settings()

        self.database = database or FaceDatabase(settings.db_path)
        self.detector = detector or build_detector(settings.detector_backend)
        self.encoder = encoder or FaceEncoder()
        self.recognizer = recognizer or FaceRecognizer(
            self.database, tolerance=settings.tolerance, metric=settings.distance_metric
        )
        # Asegura que el recognizer sepa qué personas hay.
        self.recognizer.reload()
        logger.debug("FacePipeline inicializado")

    # --- Registro -----------------------------------------------------------

    def register_person(
        self,
        name: str,
        image_paths: list[str | Path],
        notes: str = "",
        num_jitters: int = 1,
    ) -> Person:
        """Registra una persona a partir de una o más imágenes.

        Cada imagen puede contener UN rostro. Si tiene varios, se toma el más grande.

        Args:
            name: Nombre único.
            image_paths: Rutas a imágenes.
            num_jitters: Jitters al calcular el embedding.

        Raises:
            ValueError: Si ninguna contiene al menos un rostro.
        """
        person = self.database.add_person(name, notes=notes)
        registered = 0

        for path in image_paths:
            try:
                encoding = self.encode_single_face(path, num_jitters=num_jitters)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Saltando {}: {}", path, exc)
                continue

            self.database.add_sample(
                person_id=person.id,
                encoding=encoding,
                image_path=str(path),
                source="register",
            )
            registered += 1

        if registered == 0:
            # Sin muestras válidas: revertir creación.
            self.database.delete_person(person.id)
            raise ValueError(f"Ninguna imagen contenía un rostro válido para '{name}'")

        self.recognizer.mark_dirty()
        logger.success("Persona '{}' registrada con {} muestra(s)", name, registered)
        return self.database.get_person(person.id)

    def encode_single_face(
        self,
        image_path: str | Path,
        num_jitters: int = 1,
    ) -> FaceEncoding:
        """Detecta y codifica el rostro más grande de una imagen.

        Raises:
            ValueError: Si no se detecta ningún rostro.
        """
        from .utils import load_image

        image = load_image(image_path, max_size=get_settings().max_image_size)
        locations = self.detector.detect(image)

        if not locations:
            raise ValueError(f"No se detectaron rostros en {image_path}")

        # Si hay varios, tomar el más grande (proxy de "rostro principal").
        largest = max(locations, key=lambda loc: loc.area)
        encodings = self.encoder.encode(image, [largest], num_jitters=num_jitters)
        if not encodings:
            raise ValueError(f"No se pudo codificar el rostro en {image_path}")
        return encodings[0]

    # --- Reconocimiento -----------------------------------------------------

    def recognize_image(self, image_path: str | Path) -> RecognitionResult:
        """Reconoce todos los rostros en una imagen estática.

        Args:
            image_path: Ruta a la imagen.

        Returns:
            `RecognitionResult` con todas las coincidencias.
        """
        from .utils import load_image

        start = time.perf_counter()
        image = load_image(image_path, max_size=get_settings().max_image_size)
        return self.recognize_array(image, image_path=str(image_path))

    def recognize_array(
        self,
        image: np.ndarray,
        image_path: str | None = None,
    ) -> RecognitionResult:
        """Reconoce rostros sobre un arreglo RGB directamente.

        Args:
            image: ndarray (H, W, 3) RGB.
            image_path: Etiqueta opcional para incluir en el resultado.

        Returns:
            `RecognitionResult`.
        """
        start = time.perf_counter()
        locations: list[FaceLocation] = self.detector.detect(image)
        encodings: list[FaceEncoding] = self.encoder.encode(image, locations)
        matches = self.recognizer.recognize_batch(encodings)
        elapsed = (time.perf_counter() - start) * 1000.0

        h, w = image.shape[:2]
        return RecognitionResult(
            image_path=image_path,
            faces=matches,
            image_width=w,
            image_height=h,
            elapsed_ms=elapsed,
        )

    # --- Webcam -------------------------------------------------------------

    def iter_webcam(
        self,
        camera_index: int = 0,
        skip_frames: int = 1,
        max_frames: int | None = None,
    ) -> Iterator[RecognitionResult]:
        """Generador que procesa frames de la webcam en vivo.

        Args:
            camera_index: Índice de la cámara (0 por defecto).
            skip_frames: Procesar 1 de cada N frames (mayor = más FPS).
            max_frames: Máximo de frames a procesar (None = infinito).

        Yields:
            `RecognitionResult` por cada frame procesado.
        """
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("opencv-python es requerido para webcam") from exc

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {camera_index}")

        logger.info("Webcam {} abierta ({}x{})", camera_index, int(cap.get(3)), int(cap.get(4)))

        frame_idx = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    logger.warning("Frame no leído, saliendo del loop")
                    break

                if frame_idx % skip_frames == 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    yield self.recognize_array(rgb, image_path=f"webcam:{frame_idx}")

                    if max_frames is not None and frame_idx // skip_frames >= max_frames:
                        break

                frame_idx += 1
        finally:
            cap.release()
            logger.info("Webcam liberada")

    def draw_results(
        self,
        image: np.ndarray,
        result: RecognitionResult,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """Dibuja bounding boxes y etiquetas sobre una imagen (devuelve BGR).

        Útil para visualizar el reconocimiento sobre imágenes estáticas o webcam.
        """
        import cv2

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image.copy()
        for match in result.faces:
            loc = match.encoding.location
            if loc is None:
                continue
            color = (0, 255, 0) if match.matched else (0, 0, 255)
            cv2.rectangle(bgr, (loc.left, loc.top), (loc.right, loc.bottom), color, 2)

            label = match.person.name if match.matched else "Desconocido"
            if show_confidence and match.matched:
                label = f"{label} ({match.confidence:.0%})"

            # Fondo del label para que se vea sobre cualquier imagen.
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                bgr,
                (loc.left, loc.top - th - 8),
                (loc.left + tw, loc.top),
                color,
                -1,
            )
            cv2.putText(
                bgr,
                label,
                (loc.left, loc.top - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        return bgr

    def close(self) -> None:
        """Cierra recursos (db)."""
        self.database.close()

    def __enter__(self) -> "FacePipeline":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


__all__ = ["FacePipeline", "RecognitionResult"]