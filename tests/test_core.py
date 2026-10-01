"""Tests unitarios — no requieren red ni cámara."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from reconocimiento_facial.config import reload_settings
from reconocimiento_facial.database import FaceDatabase, Person
from reconocimiento_facial.encoder import FaceEncoding
from reconocimiento_facial.recognizer import FaceRecognizer


@pytest.fixture
def tmp_db(tmp_path: Path) -> FaceDatabase:
    """Crea una DB temporal por test."""
    db_path = tmp_path / "test.db"
    db = FaceDatabase(db_path)
    yield db
    db.close()


@pytest.fixture
def tmp_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirige la config global a un directorio temporal."""
    monkeypatch.setenv("RF_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RF_DB_PATH", "test.db")
    reload_settings()


def make_encoding(seed: int) -> FaceEncoding:
    """Genera un encoding determinista (128-d) con una semilla."""
    rng = np.random.default_rng(seed)
    return FaceEncoding(vector=rng.standard_normal(128).astype(np.float64))


class TestFaceEncoding:
    """Tests del dataclass `FaceEncoding`."""

    def test_shape_validation(self) -> None:
        """Solo se aceptan vectores de 128 floats."""
        with pytest.raises(ValueError, match="forma"):
            FaceEncoding(vector=np.zeros(64, dtype=np.float64))

    def test_to_list_round_trip(self) -> None:
        """Serializa y reconstruye sin pérdida."""
        original = make_encoding(42)
        as_list = original.to_list()
        assert len(as_list) == 128
        recovered = FaceEncoding.from_list(as_list)
        np.testing.assert_array_equal(original.vector, recovered.vector)

    def test_immutability(self) -> None:
        """El vector no se puede modificar tras la creación."""
        enc = make_encoding(1)
        with pytest.raises(Exception):  # FrozenInstanceError
            enc.vector = np.zeros(128)  # type: ignore[misc]


class TestFaceDatabase:
    """Tests de la base de datos."""

    def test_add_and_get_person(self, tmp_db: FaceDatabase) -> None:
        person = tmp_db.add_person("Ana")
        fetched = tmp_db.get_person(person.id)
        assert fetched.name == "Ana"
        assert fetched.sample_count == 0

    def test_add_person_idempotent(self, tmp_db: FaceDatabase) -> None:
        """Añadir el mismo nombre dos veces devuelve el mismo ID."""
        first = tmp_db.add_person("Ana")
        second = tmp_db.add_person("Ana")
        assert first.id == second.id

    def test_get_person_not_found(self, tmp_db: FaceDatabase) -> None:
        with pytest.raises(KeyError):
            tmp_db.get_person("inexistente-id")

    def test_list_persons(self, tmp_db: FaceDatabase) -> None:
        tmp_db.add_person("Ana")
        tmp_db.add_person("Beto")
        tmp_db.add_person("Carla")
        persons = tmp_db.list_persons()
        assert {p.name for p in persons} == {"Ana", "Beto", "Carla"}

    def test_add_sample_increases_count(self, tmp_db: FaceDatabase) -> None:
        person = tmp_db.add_person("Ana")
        enc = make_encoding(7)
        tmp_db.add_sample(person.id, enc)
        tmp_db.add_sample(person.id, make_encoding(8))
        assert tmp_db.count_samples(person.id) == 2

    def test_add_sample_unknown_person(self, tmp_db: FaceDatabase) -> None:
        with pytest.raises(KeyError):
            tmp_db.add_sample("no-existe", make_encoding(1))

    def test_delete_person_removes_samples(self, tmp_db: FaceDatabase) -> None:
        person = tmp_db.add_person("Ana")
        tmp_db.add_sample(person.id, make_encoding(5))
        assert tmp_db.delete_person(person.id) is True
        assert tmp_db.count_samples() == 0
        assert tmp_db.delete_person(person.id) is False  # ya no existe

    def test_iter_all_for_training(self, tmp_db: FaceDatabase) -> None:
        ana = tmp_db.add_person("Ana")
        beto = tmp_db.add_person("Beto")
        tmp_db.add_sample(ana.id, make_encoding(1))
        tmp_db.add_sample(beto.id, make_encoding(2))
        tmp_db.add_sample(beto.id, make_encoding(3))

        result = {p.name: len(encs) for p, encs in tmp_db.iter_all_for_training()}
        assert result == {"Ana": 1, "Beto": 2}


class TestFaceRecognizer:
    """Tests del reconocedor."""

    def test_empty_database(self, tmp_db: FaceDatabase) -> None:
        """Sin muestras, todo es Desconocido."""
        rec = FaceRecognizer(tmp_db, tolerance=0.6)
        rec.reload()
        match = rec.recognize(make_encoding(42))
        assert not match.matched
        assert match.person is None

    def test_exact_match(self, tmp_db: FaceDatabase) -> None:
        """Un encoding idéntico debe coincidir."""
        person = tmp_db.add_person("Ana")
        enc = make_encoding(123)
        tmp_db.add_sample(person.id, enc)
        rec = FaceRecognizer(tmp_db, tolerance=0.6)
        rec.reload()

        match = rec.recognize(enc)
        assert match.matched is True
        assert match.person is not None
        assert match.person.id == person.id
        assert match.distance < 0.01

    def test_distance_threshold(self, tmp_db: FaceDatabase) -> None:
        """Respetar tolerancia: un rostro lejano queda como Desconocido."""
        person = tmp_db.add_person("Ana")
        tmp_db.add_sample(person.id, make_encoding(1))
        rec = FaceRecognizer(tmp_db, tolerance=0.05)  # muy estricto
        rec.reload()

        # Encoding muy diferente (semilla lejana).
        match = rec.recognize(make_encoding(999))
        assert match.matched is False

    def test_invalid_tolerance(self, tmp_db: FaceDatabase) -> None:
        with pytest.raises(ValueError):
            FaceRecognizer(tmp_db, tolerance=1.5)

    def test_invalid_metric(self, tmp_db: FaceDatabase) -> None:
        with pytest.raises(ValueError):
            FaceRecognizer(tmp_db, metric="manhattan")

    def test_candidates_ranked(self, tmp_db: FaceDatabase) -> None:
        """Las personas más cercanas salen primero."""
        ana = tmp_db.add_person("Ana")
        beto = tmp_db.add_person("Beto")
        # Usamos el MISMO encoding para Ana, otro para Beto.
        ana_enc = make_encoding(1)
        beto_enc = make_encoding(2)
        tmp_db.add_sample(ana.id, ana_enc)
        tmp_db.add_sample(beto.id, beto_enc)

        rec = FaceRecognizer(tmp_db, tolerance=0.6)
        rec.reload()

        # El query es el de Ana, así que Ana debe estar más cerca.
        match = rec.recognize(ana_enc)
        assert match.candidates[0][0].id == ana.id
        # Y la distancia con Ana debe ser menor que con Beto.
        ana_dist = match.candidates[0][1]
        beto_dist = next(d for (p, d) in match.candidates if p.id == beto.id)
        assert ana_dist < beto_dist


class TestUtils:
    """Tests de utilidades."""

    def test_encode_decode_b64_round_trip(self) -> None:
        from reconocimiento_facial.utils import decode_image_b64, encode_image_b64

        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
        encoded = encode_image_b64(img)
        decoded = decode_image_b64(encoded)
        np.testing.assert_array_equal(img, decoded)

    def test_decode_b64_with_data_uri(self) -> None:
        from reconocimiento_facial.utils import decode_image_b64

        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
        from reconocimiento_facial.utils import encode_image_b64

        b64 = encode_image_b64(img)
        data_uri = f"data:image/png;base64,{b64}"
        decoded = decode_image_b64(data_uri)
        np.testing.assert_array_equal(img, decoded)


class TestConfig:
    """Tests de configuración."""

    def test_default_db_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RF_DATA_DIR", raising=False)
        monkeypatch.delenv("RF_DB_PATH", raising=False)
        s = reload_settings()
        # La ruta absoluta se construye en post_init.
        assert s.db_path.is_absolute()

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("RF_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("RF_TOLERANCE", "0.4")
        monkeypatch.setenv("RF_DETECTOR_BACKEND", "opencv_haar")
        s = reload_settings()
        assert s.tolerance == 0.4
        assert s.detector_backend == "opencv_haar"

    def test_invalid_tolerance_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RF_TOLERANCE", "1.5")
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            reload_settings()


@pytest.mark.integration
class TestAPI:
    """Tests de la API (usan TestClient pero no cámara)."""

    def test_health_endpoint(self, tmp_settings: None, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        from reconocimiento_facial.api import create_app

        client = TestClient(create_app())
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "database" in body

    def test_create_person(self, tmp_settings: None) -> None:
        from fastapi.testclient import TestClient

        from reconocimiento_facial.api import create_app

        client = TestClient(create_app())
        response = client.post(
            "/persons",
            json={"name": "Test User", "notes": "test"},
        )
        assert response.status_code == 201, response.text
        data = response.json()
        assert data["name"] == "Test User"
        assert data["sample_count"] == 0

    def test_list_persons(self, tmp_settings: None) -> None:
        from fastapi.testclient import TestClient

        from reconocimiento_facial.api import create_app

        client = TestClient(create_app())
        client.post("/persons", json={"name": "Alice"})
        client.post("/persons", json={"name": "Bob"})

        response = client.get("/persons")
        assert response.status_code == 200
        names = {p["name"] for p in response.json()}
        assert names == {"Alice", "Bob"}