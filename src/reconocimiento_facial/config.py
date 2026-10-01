"""Configuración global del sistema."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


DetectorBackend = Literal["face_recognition", "opencv_haar", "opencv_dnn"]
DistanceMetric = Literal["euclidean", "euclidean_l2"]


class Settings(BaseSettings):
    """Configuración cargada desde variables de entorno o `.env`.

    Attributes:
        data_dir: Directorio base para datos persistentes (DB, imágenes).
        db_path:  Ruta del archivo SQLite. Si es relativa, se resuelve contra `data_dir`.
        tolerance: Umbral de distancia para considerar una coincidencia (0.0–1.0).
                   Menor valor = más estricto. Por defecto 0.6 (estándar de face_recognition).
        detector_backend: Backend de rostros a usar.
        distance_metric: Métrica de distancia entre embeddings.
        max_image_size: Tamaño máximo (largo) al que se redimensionan imágenes para acelerar
                        el procesamiento. None desactiva la redimensión.
        log_level: Nivel de logging.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RF_",
        case_sensitive=False,
        extra="ignore",
    )

    data_dir: Path = Field(default_factory=lambda: Path("./data"))
    db_path: Path = Field(default_factory=lambda: Path("reconocimiento.db"))

    tolerance: float = Field(default=0.6, ge=0.0, le=1.0)
    detector_backend: DetectorBackend = Field(default="face_recognition")
    distance_metric: DistanceMetric = Field(default="euclidean_l2")

    max_image_size: int | None = Field(default=800, ge=100)
    log_level: str = Field(default="INFO")

    def model_post_init(self, __context) -> None:  # type: ignore[no-untyped-def]
        """Crea directorios y resuelve rutas relativas después de la validación."""
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if not self.db_path.is_absolute():
            self.db_path = (self.data_dir / self.db_path).resolve()


_settings: Settings | None = None


def get_settings() -> Settings:
    """Devuelve la instancia singleton de configuración.

    Se crea de forma lazy para que `Settings` se evalúe al primer uso, no al importar.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Recarga la configuración desde disco (útil en tests)."""
    global _settings
    _settings = Settings()
    return _settings