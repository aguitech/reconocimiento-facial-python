"""API REST con FastAPI.

Endpoints:
    GET  /health                      Estado del servicio.
    POST /persons                     Crea una persona (sin imágenes).
    POST /persons/register            Crea una persona + registra imágenes.
    GET  /persons                     Lista todas las personas.
    GET  /persons/{id}                Detalle de una persona.
    DELETE /persons/{id}              Elimina una persona.
    POST /recognize                   Reconoce rostros en una imagen.
    POST /recognize/file              Reconoce rostros en un archivo subido (multipart).
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import get_settings
from .pipeline import FacePipeline, RecognitionResult
from .recognizer import Match
from .schemas import (
    FaceSampleOut,
    HealthResponse,
    PersonCreate,
    PersonOut,
    PersonRegisterWithImages,
    RecognizeResponse,
)
from .utils import configure_logging, decode_image_b64, logger


def create_app(pipeline: FacePipeline | None = None) -> FastAPI:
    """Crea la aplicación FastAPI.

    Args:
        pipeline: Pipeline inyectado. Si None, se construye uno al inicio (lazy).
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Reconocimiento Facial API",
        description="API REST para detectar y reconocer rostros en imágenes.",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS abierto por defecto (API interna). En producción, restringir orígenes.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _pipeline: FacePipeline | None = pipeline

    def get_pipeline() -> FacePipeline:
        """Inicializa el pipeline lazy en el primer request."""
        nonlocal _pipeline
        if _pipeline is None:
            _pipeline = FacePipeline()
        return _pipeline

    # --- Helpers -------------------------------------------------------------

    def match_to_sample_out(match: Match) -> FaceSampleOut:
        """Convierte un `Match` interno al schema de salida."""
        loc = match.encoding.location
        location_dict = {
            "top": loc.top if loc else 0,
            "right": loc.right if loc else 0,
            "bottom": loc.bottom if loc else 0,
            "left": loc.left if loc else 0,
        }
        return FaceSampleOut(
            matched=match.matched,
            person_id=match.person.id if match.matched else None,
            person_name=match.person.name if match.matched else None,
            distance=float(match.distance) if not np.isnan(match.distance) else -1.0,
            confidence=match.confidence,
            location=location_dict,
            top_candidates=[
                {"person_id": str(p.id), "person_name": p.name, "distance": float(d)}
                for (p, d) in match.candidates
            ],
        )

    def person_to_out(person) -> PersonOut:
        return PersonOut(
            id=person.id,
            name=person.name,
            notes=person.notes,
            metadata=person.metadata,
            created_at=person.created_at,
            sample_count=person.sample_count,
        )

    def result_to_response(result: RecognitionResult) -> RecognizeResponse:
        return RecognizeResponse(
            image_width=result.image_width,
            image_height=result.image_height,
            elapsed_ms=result.elapsed_ms,
            faces=[match_to_sample_out(m) for m in result.faces],
        )

    # --- Rutas ---------------------------------------------------------------

    @app.get("/", response_model=dict)
    async def root() -> dict:
        return {"message": "Reconocimiento Facial API", "version": __version__, "docs": "/docs"}

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Estado del servicio."""
        try:
            p = get_pipeline()
            persons = p.database.list_persons()
            samples = p.database.count_samples()
            return HealthResponse(
                status="ok",
                version=__version__,
                database={
                    "persons": len(persons),
                    "samples": samples,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Health check falló")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Servicio no disponible: {exc}",
            ) from exc

    @app.post("/persons", response_model=PersonOut, status_code=status.HTTP_201_CREATED)
    async def create_person(payload: PersonCreate) -> PersonOut:
        """Crea una persona sin registrar imágenes todavía."""
        p = get_pipeline()
        try:
            person = p.database.add_person(
                name=payload.name,
                notes=payload.notes,
                metadata=payload.metadata,
            )
            p.recognizer.mark_dirty()
            return person_to_out(person)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error creando persona")
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc

    @app.post(
        "/persons/register",
        response_model=PersonOut,
        status_code=status.HTTP_201_CREATED,
    )
    async def register_person(payload: PersonRegisterWithImages) -> PersonOut:
        """Crea una persona y registra imágenes (en base64) como muestras."""
        p = get_pipeline()
        temp_paths: list[Path] = []
        try:
            # Decodificar todas las imágenes a archivos temporales.
            for idx, b64 in enumerate(payload.images):
                img_array = decode_image_b64(b64)
                # Guardar como PNG temporal.
                tmp = Path(tempfile.gettempdir()) / f"rf_register_{datetime.now().timestamp()}_{idx}.png"
                from PIL import Image

                Image.fromarray(img_array).save(tmp)
                temp_paths.append(tmp)

            person = p.register_person(
                name=payload.name,
                image_paths=temp_paths,
                notes=payload.notes,
            )
            return person_to_out(person)
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error registrando persona")
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc
        finally:
            for path in temp_paths:
                try:
                    path.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001, S110
                    pass

    @app.get("/persons", response_model=list[PersonOut])
    async def list_persons() -> list[PersonOut]:
        """Lista todas las personas."""
        p = get_pipeline()
        persons = p.database.list_persons()
        return [person_to_out(person) for person in persons]

    @app.get("/persons/{person_id}", response_model=PersonOut)
    async def get_person(person_id: str) -> PersonOut:
        """Detalle de una persona por ID."""
        p = get_pipeline()
        try:
            person = p.database.get_person(person_id)
            return person_to_out(person)
        except KeyError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    @app.delete("/persons/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_person(person_id: str) -> None:
        """Elimina una persona."""
        p = get_pipeline()
        if not p.database.delete_person(person_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Persona no encontrada")
        p.recognizer.mark_dirty()

    @app.post("/recognize", response_model=RecognizeResponse)
    async def recognize(body: dict[str, Any]) -> RecognizeResponse:
        """Reconoce rostros en una imagen enviada en JSON.

        Body esperado:
            {
                "image": "<base64 o data URI>"
            }
        """
        if "image" not in body:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Falta 'image' en el body")

        try:
            img = decode_image_b64(body["image"])
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

        p = get_pipeline()
        result = p.recognize_array(img)
        return result_to_response(result)

    @app.post("/recognize/file", response_model=RecognizeResponse)
    async def recognize_file(file: UploadFile = File(...)) -> RecognizeResponse:
        """Reconoce rostros en un archivo subido (multipart/form-data)."""
        from PIL import Image
        from io import BytesIO

        try:
            contents = await file.read()
            with Image.open(BytesIO(contents)) as im:
                img_array = np.array(im.convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"No se pudo leer la imagen: {exc}"
            ) from exc

        p = get_pipeline()
        result = p.recognize_array(img_array)
        return result_to_response(result)

    return app


# Instancia para `uvicorn reconocimiento_facial.api:app`.
app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "reconocimiento_facial.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )