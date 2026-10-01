"""Base de datos de personas conocidas.

Modelos:
    - Person: Persona registrada con nombre y metadatos.
    - FaceSample: Muestra (encoding + ruta de imagen) asociada a una persona.

Almacenamiento:
    - SQLite en `data/reconocimiento.db`.
    - Los encodings se guardan como JSON en TEXT (suficiente para 128 floats).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

import numpy as np

from .encoder import FaceEncoding
from .utils import logger


@dataclass
class Person:
    """Persona registrada en el sistema."""

    id: str
    name: str
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sample_count: int = 0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Person":
        """Construye un `Person` desde una fila de SQLite."""
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        return cls(
            id=row["id"],
            name=row["name"],
            notes=row["notes"] or "",
            metadata=metadata,
            created_at=datetime.fromisoformat(row["created_at"]),
            sample_count=row["sample_count"] or 0,
        )


class FaceDatabase:
    """Base de datos SQLite thread-safe de personas y muestras."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS persons (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        notes TEXT DEFAULT '',
        metadata TEXT DEFAULT '{}',
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS face_samples (
        id TEXT PRIMARY KEY,
        person_id TEXT NOT NULL,
        encoding TEXT NOT NULL,
        image_path TEXT DEFAULT '',
        source TEXT DEFAULT '',
        created_at TEXT NOT NULL,
        FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_samples_person ON face_samples(person_id);
    CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name);
    """

    def __init__(self, db_path) -> None:  # type: ignore[no-untyped-def]
        """Inicializa la DB y crea las tablas si no existen."""
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._connect()
        self._create_schema()
        logger.debug("FaceDatabase inicializada en {}", db_path)

    # --- Gestión de conexión -------------------------------------------------

    def _connect(self) -> None:
        """Abre la conexión SQLite y configura PRAGMAs de rendimiento."""
        self._conn = sqlite3.connect(
            str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
            isolation_level=None,  # autocommit; usamos BEGIN explícito.
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.execute("PRAGMA journal_mode = WAL;")

    def _create_schema(self) -> None:
        """Crea las tablas requeridas."""
        assert self._conn is not None
        with self._lock:
            self._conn.executescript(self.SCHEMA)

    def close(self) -> None:
        """Cierra la conexión a la base de datos."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> "FaceDatabase":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # --- CRUD de personas ----------------------------------------------------

    def add_person(self, name: str, notes: str = "", metadata: dict | None = None) -> Person:
        """Crea una nueva persona. Si ya existe una con ese nombre, la devuelve.

        Args:
            name: Nombre único.
            notes: Notas opcionales.
            metadata: Diccionario libre (p. ej. {"email": "..."}).

        Returns:
            `Person` creada o existente.
        """
        assert self._conn is not None
        with self._lock:
            existing = self._conn.execute(
                "SELECT id FROM persons WHERE name = ?", (name,)
            ).fetchone()
            if existing:
                return self.get_person(existing["id"])  # type: ignore[return-value]

            person_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                """
                INSERT INTO persons (id, name, notes, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (person_id, name, notes, json.dumps(metadata or {}), now),
            )
            logger.info("Persona creada: {} ({})", name, person_id)
            return Person(
                id=person_id,
                name=name,
                notes=notes,
                metadata=metadata or {},
                created_at=datetime.fromisoformat(now),
            )

    def get_person(self, person_id: str) -> Person:
        """Devuelve una persona por ID.

        Raises:
            KeyError: Si no existe.
        """
        assert self._conn is not None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT p.*, COUNT(s.id) AS sample_count
                FROM persons p
                LEFT JOIN face_samples s ON s.person_id = p.id
                WHERE p.id = ?
                GROUP BY p.id
                """,
                (person_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Persona no encontrada: {person_id}")
            return Person.from_row(row)

    def get_person_by_name(self, name: str) -> Person | None:
        """Devuelve una persona por nombre o None si no existe."""
        assert self._conn is not None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT p.*, COUNT(s.id) AS sample_count
                FROM persons p
                LEFT JOIN face_samples s ON s.person_id = p.id
                WHERE p.name = ?
                GROUP BY p.id
                """,
                (name,),
            ).fetchone()
            return Person.from_row(row) if row else None

    def list_persons(self) -> list[Person]:
        """Lista todas las personas registradas."""
        assert self._conn is not None
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT p.*, COUNT(s.id) AS sample_count
                FROM persons p
                LEFT JOIN face_samples s ON s.person_id = p.id
                GROUP BY p.id
                ORDER BY p.created_at DESC
                """,
            ).fetchall()
            return [Person.from_row(row) for row in rows]

    def delete_person(self, person_id: str) -> bool:
        """Elimina una persona y todas sus muestras.

        Returns:
            True si se eliminó, False si no existía.
        """
        assert self._conn is not None
        with self._lock:
            cur = self._conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            deleted = cur.rowcount > 0
            if deleted:
                logger.info("Persona eliminada: {}", person_id)
            return deleted

    # --- Muestras de rostros --------------------------------------------------

    def add_sample(
        self,
        person_id: str,
        encoding: FaceEncoding,
        image_path: str = "",
        source: str = "",
    ) -> str:
        """Agrega una muestra (encoding) a una persona.

        Args:
            person_id: ID de la persona.
            encoding: `FaceEncoding` calculado.
            image_path: Ruta de la imagen original (opcional).
            source: Fuente (ej. "webcam", "upload").

        Returns:
            ID de la muestra creada.

        Raises:
            KeyError: Si la persona no existe.
        """
        assert self._conn is not None
        with self._lock:
            # Verificar que la persona existe (FK falla silenciosamente en algunos casos).
            row = self._conn.execute("SELECT id FROM persons WHERE id = ?", (person_id,)).fetchone()
            if row is None:
                raise KeyError(f"Persona no encontrada: {person_id}")

            sample_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                """
                INSERT INTO face_samples (id, person_id, encoding, image_path, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    person_id,
                    json.dumps(encoding.to_list()),
                    image_path,
                    source,
                    now,
                ),
            )
            logger.debug("Muestra agregada a {}: {}", person_id, sample_id)
            return sample_id

    def get_samples(self, person_id: str | None = None) -> list[FaceEncoding]:
        """Devuelve todas las muestras de una persona (o de todas si `person_id` es None)."""
        assert self._conn is not None
        with self._lock:
            if person_id:
                rows = self._conn.execute(
                    "SELECT encoding FROM face_samples WHERE person_id = ?",
                    (person_id,),
                ).fetchall()
            else:
                rows = self._conn.execute("SELECT encoding FROM face_samples").fetchall()

            encodings: list[FaceEncoding] = []
            for row in rows:
                vec = np.array(json.loads(row["encoding"]), dtype=np.float64)
                encodings.append(FaceEncoding(vector=vec))
            return encodings

    def count_samples(self, person_id: str | None = None) -> int:
        """Cuenta muestras."""
        assert self._conn is not None
        with self._lock:
            if person_id:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS c FROM face_samples WHERE person_id = ?",
                    (person_id,),
                ).fetchone()
            else:
                row = self._conn.execute("SELECT COUNT(*) AS c FROM face_samples").fetchone()
            return int(row["c"])  # type: ignore[arg-type]

    # --- Bulk ----------------------------------------------------------------

    def iter_all_for_training(self) -> Iterator[tuple[Person, list[FaceEncoding]]]:
        """Itera (persona, [encodings]) sobre toda la base. Útil para entrenar.

        Yields:
            Tuplas `(Person, [FaceEncoding])` para cada persona con muestras.
        """
        for person in self.list_persons():
            samples = self.get_samples(person.id)
            if samples:
                yield person, samples


__all__ = ["FaceDatabase", "Person"]