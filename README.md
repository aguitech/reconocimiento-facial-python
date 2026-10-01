# Reconocimiento Facial — Python

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)

Sistema de **reconocimiento facial** en Python con detección, encodings, base de datos persistente, API REST (FastAPI), CLI (Typer) y soporte para webcam. Listo para correr en local, en servidor, o en la nube vía Docker.

## ✨ Características

- 🧠 **Detección dual**: `face_recognition` (dlib) o Haar cascades de OpenCV.
- 🎯 **Embeddings 128-d** usando el modelo dlib preentrenado.
- 💾 **Base de datos SQLite** con personas y muestras (encoding + metadatos).
- ⚡ **Pipeline de alto nivel** (`FacePipeline`) que orquesta todo.
- 🌐 **API REST con FastAPI**: registro, listado, eliminación, reconocimiento.
- 🖥️ **CLI con Typer**: `register`, `recognize`, `list`, `delete`, `serve`, etc.
- 🎥 **Webcam en vivo** con visualización.
- 🐳 **Docker multi-stage** listo para producción.
- 🧪 **Tests con pytest** (cubren core, API y utilidades).

## 📦 Requisitos

- Python 3.9+
- macOS, Linux o Windows
- ~200 MB para dependencias (incluye dlib compilado)

## 🚀 Instalación rápida

### Con `pip` (modo desarrollo)

```bash
git clone git@github.com:aguitech/reconocimiento-facial-python.git
cd reconocimiento-facial-python
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Con Docker

```bash
docker-compose up --build
# API disponible en http://localhost:8000/docs
```

## 🎯 Uso

### CLI — registrar rostros

```bash
# Registrar a "Ana" con varias imágenes
rf-detect register person --name "Ana" \
    --image fotos/ana_1.jpg \
    --image fotos/ana_2.jpg \
    --image fotos/ana_3.jpg \
    --notes "Compañera de trabajo" \
    --jitters 10
```

### CLI — reconocer en una imagen

```bash
rf-detect recognize image fotos/grupo.jpg --output resultado.png --show
```

### CLI — webcam en vivo

```bash
rf-detect recognize webcam --camera 0 --skip 2
```

### CLI — listar y eliminar

```bash
rf-detect list
rf-detect delete <id-parcial>
rf-detect stats
rf-detect serve --host 0.0.0.0 --port 8000
```

### API con `curl`

```bash
# Crear persona
curl -X POST http://localhost:8000/persons \
    -H "Content-Type: application/json" \
    -d '{"name":"Ana","notes":"CEO"}'

# Registrar con imágenes base64
curl -X POST http://localhost:8000/persons/register \
    -H "Content-Type: application/json" \
    -d '{"name":"Beto","images":["<BASE64>", ...]}'

# Reconocer
curl -X POST http://localhost:8000/recognize/file \
    -F "file=@t Foto.jpg"

# Eliminar
curl -X DELETE http://localhost:8000/persons/<id>
```

### Python — uso programático

```python
from reconocimiento_facial import FacePipeline

with FacePipeline() as pipeline:
    # Registrar
    pipeline.register_person(
        name="Ana",
        image_paths=["fotos/ana_1.jpg", "fotos/ana_2.jpg"],
    )

    # Reconocer
    result = pipeline.recognize_image("fotos/grupo.jpg")
    for match in result.faces:
        name = match.person.name if match.matched else "Desconocido"
        print(f"{name}: distancia={match.distance:.3f}, confianza={match.confidence:.0%}")

    # Webcam
    for frame_result in pipeline.iter_webcam(camera_index=0):
        print(f"Rostros: {len(frame_result.faces)}")
```

## ⚙️ Configuración

Variables de entorno (todas con prefijo `RF_`):

| Variable | Default | Descripción |
|---|---|---|
| `RF_DATA_DIR` | `./data` | Carpeta de datos persistentes |
| `RF_DB_PATH` | `reconocimiento.db` | Archivo de base de datos (relativo a `DATA_DIR`) |
| `RF_TOLERANCE` | `0.6` | Umbral de coincidencia (0.0–1.0, menor = más estricto) |
| `RF_DETECTOR_BACKEND` | `face_recognition` | `face_recognition` u `opencv_haar` |
| `RF_DISTANCE_METRIC` | `euclidean_l2` | `euclidean` o `euclidean_l2` |
| `RF_MAX_IMAGE_SIZE` | `800` | Redimensionar imágenes (None desactiva) |
| `RF_LOG_LEVEL` | `default` | Nivel de loguru (TRACE, DEBUG, INFO, WARNING, ERROR) |

Ejemplo `.env`:

```ini
RF_TOLERANCE=0.5
RF_DETECTOR_BACKEND=face_recognition
RF_LOG_LEVEL=INFO
```

## 🧪 Tests

```bash
pytest                       # corre todo
pytest -m "not integration"  # sin integración (sin servidor)
pytest --cov=reconocimiento_facial
```

## 🏗️ Arquitectura

```
src/reconocimiento_facial/
├── __init__.py
├── config.py          # Settings (env + .env)
├── utils.py           # logging, load/encode/decode de imágenes
├── detector.py        # FaceLocation + FaceRecognitionDetector + OpenCVHaarDetector
├── encoder.py         # FaceEncoding + FaceEncoder (128-d embeddings)
├── database.py        # FaceDatabase (SQLite) + Person
├── recognizer.py      # FaceRecognizer + Match
├── pipeline.py        # FacePipeline (alto nivel)
├── schemas.py         # Modelos Pydantic para la API
├── api.py             # FastAPI (crear/listar/eliminar/reconocer)
└── cli.py              # Typer (rf-detect CLI)
```

### Flujo de reconocimiento

```
imagen (RGB, uint8)
    │
    ▼
[Detector]      → list[FaceLocation]      (bounding boxes)
    │
    ▼
[Encoder]       → list[FaceEncoding]     (vectores 128-d)
    │
    ▼
[Recognizer]    → list[Match]            (persona + distancia + confianza)
    │
    ▼
Reconocimiento ✓
```

## 📊 Benchmarks (referencia, MacBook Pro M2)

| Operación | Tiempo medio |
|---|---|
| Detección HOG (1280×720) | ~85 ms |
| Encoding 1 rostro | ~25 ms |
| Encoding 1 rostro (jitters=10) | ~220 ms |
| Comparación contra 100 muestras | <1 ms |
| Reconocimiento completo (1 rostro, 10 muestras) | ~115 ms |

## 🗺️ Roadmap

- [x] Detector dual (dlib + Haar)
- [x] Pipeline de alto nivel
- [x] Persistencia SQLite
- [x] API REST
- [x] CLI
- [x] Docker
- [ ] Aceleración con GPU (CUDA / CoreML)
- [ ] Reconocimiento con anti-spoofing (liveness detection)
- [ ] Cluster de embeddings (DBSCAN) para entrenar a partir de muchos rostros
- [ ] UI web (Streamlit)
- [ ] Integración con Kafka / RabbitMQ para eventos

## 🤝 Contribuciones

¡PRs bienvenidos! Antes de abrir uno:

```bash
ruff check src tests
black src tests
mypy src
pytest
```

## 📝 Licencia

MIT — ver [LICENSE](LICENSE).

## ✍️ Autor

**Héctor Aguilar** — [aguitech](https://github.com/aguitech) — hector@aguitech.com