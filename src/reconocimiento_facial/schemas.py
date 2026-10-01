"""Modelos Pydantic para la API REST."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PersonCreate(BaseModel):
    """Cuerpo para registrar una persona."""

    name: str = Field(min_length=1, max_length=120, description="Nombre único")
    notes: str = Field(default="", max_length=500)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PersonRegisterWithImages(PersonCreate):
    """Registro de persona con imágenes en base64."""

    images: list[str] = Field(
        min_length=1,
        max_length=20,
        description="Imágenes en base64 (data URI o puro). Máximo 20.",
    )

    @field_validator("images")
    @classmethod
    def strip_data_uris(cls, v: list[str]) -> list[str]:
        """Acepta tanto `data:image/...;base64,XXXX` como base64 puro."""
        cleaned = []
        for img in v:
            if img.startswith("data:"):
                img = img.split(",", 1)[1]
            cleaned.append(img)
        return cleaned


class PersonOut(BaseModel):
    """Salida con info de una persona."""

    id: str
    name: str
    notes: str
    metadata: dict[str, Any]
    created_at: datetime
    sample_count: int


class FaceSampleOut(BaseModel):
    """Salida de un rostro detectado en una imagen."""

    matched: bool
    person_id: str | None
    person_name: str | None
    distance: float
    confidence: float
    location: dict[str, int]
    top_candidates: list[dict[str, Any]]


class RecognizeResponse(BaseModel):
    """Respuesta de reconocimiento sobre una imagen."""

    image_width: int
    image_height: int
    elapsed_ms: float
    faces: list[FaceSampleOut]


class HealthResponse(BaseModel):
    """Estado del servicio."""

    status: str
    version: str
    database: dict[str, int]


__all__ = [
    "PersonCreate",
    "PersonRegisterWithImages",
    "PersonOut",
    "FaceSampleOut",
    "RecognizeResponse",
    "HealthResponse",
]