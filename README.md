<p align="center">
  <img src="https://aguitech.com/images/logo.png" alt="AGUITECH" width="120">
</p>

<h1 align="center">👤 reconocimiento-facial-python</h1>

<p align="center">
  <strong>Reconocimiento facial en Python · OpenCV, face_recognition, dlib y DeepFace.</strong><br>
  Toolkit educativo para detección, reconocimiento, verificación y análisis
  de rostros — desde cero hasta producción.
</p>

<p align="center">
  <a href="#que-es">Qué es</a> ·
  <a href="#features">Features</a> ·
  <a href="#stack">Stack</a> ·
  <a href="#instalacion">Instalación</a> ·
  <a href="#uso">Uso</a> ·
  <a href="#ejemplos">Ejemplos</a> ·
  <a href="#api">API</a>
</p>

---

## ¿Qué es

`reconocimiento-facial-python` es un **toolkit modular de reconocimiento
facial** escrito en Python. Reúne en un solo lugar las librerías más
usadas del ecosistema (OpenCV, `face_recognition`/`dlib`, `DeepFace`,
`MediaPipe`, `insightface`) detrás de una **API única y consistente**.

### Casos de uso

- 🎯 **Detección** de caras en fotos / video (bounding boxes + landmarks).
- 🆔 **Identificación** (1:N) — "¿quién es este rostro?" contra una base de datos.
- ✅ **Verificación** (1:1) — "¿esta persona es quien dice ser?"
- 😊 **Análisis de atributos** — edad estimada, género, emoción, etnia.
- 📐 **Landmarks** — 5 / 68 / 468 puntos clave del rostro.
- 🎬 **Tracking en video** — seguir un rostro a lo largo de un clip.
- 📦 **Indexado y búsqueda** — guardar embeddings en disco y consultar.

### ¿Por qué otro toolkit?

Porque cada librería tiene su propia API, formatos de entrada distintos y
resultados en estructuras diferentes. Este repo:

- **Unifica la API**: una sola interfaz `FaceEngine.detect(...)` /
  `recognize(...)` que delega al backend elegido.
- **Compara backends** lado a lado con el mismo input para ver precisión / velocidad.
- **Avanza progresivamente**: del script de 10 líneas a la API REST con workers.

## Features

| Feature | Estado | Backend |
|---------|--------|---------|
| 🔍 Detección bounding boxes | ✅ | OpenCV / DNN |
| 📐 Landmarks 5 puntos | ✅ | MediaPipe / face_recognition |
| 📐 Landmarks 68 puntos | ✅ | dlib / face_recognition |
| 📐 Landmarks 468 puntos (Face Mesh) | ✅ | MediaPipe |
| 🆔 Embeddings 128-D | ✅ | face_recognition (dlib) |
| 🆔 Embeddings 512-D (ArcFace) | ✅ | insightface |
| 🆔 Embeddings VGG-Face | ✅ | DeepFace |
| ✅ Verificación 1:1 (umbral) | ✅ | todos |
| 🆔 Identificación 1:N | ✅ | todos |
| 😊 Edad / Género / Emoción | ✅ | DeepFace |
| 🎬 Tracking en video | ✅ | SORT + centroides |
| 📦 Persistencia embeddings (pickle / HDF5) | ✅ | propio |
| 🌐 API REST (FastAPI) | 🚧 | propio |
| 🐳 Docker image | 🚧 | - |
| ⚡ Aceleración GPU (CUDA) | 📋 | opcional |
| 🛡 Liveness detection (anti-spoofing) | 📋 | MediaPipe |

✅ Implementado · 🚧 En desarrollo · 📋 Planeado

## Stack

- **Python 3.11+**
- **OpenCV** (`opencv-python`) — captura de video, DNN, image I/O.
- **face_recognition** (basado en dlib) — embeddings 128-D, landmarks 68-pt.
- **dlib** — modelos de detección y shape predictor.
- **DeepFace** — verificación multi-backend, análisis de atributos.
- **MediaPipe** — landmarks rápidos, Face Mesh.
- **insightface** *(opcional)* — embeddings ArcFace estado del arte.
- **NumPy + Pillow** — manipulación de imagen.
- **FastAPI + Uvicorn** *(opcional)* — servir como API REST.
- **pytest** — tests.

## Instalación

### 1. Requisitos del sistema

```bash
# Ubuntu / Debian — dependencias nativas de dlib y OpenCV
sudo apt-get update
sudo apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    build-essential cmake \
    libopenblas-dev liblapack-dev \
    libboost-all-dev \
    libgl1 libglib2.0-0
```

> **Windows / macOS**: dlib pre-compilado se instala con `pip install dlib`
> directo. Para compilar desde fuente en Windows necesitas Visual Studio
> Build Tools 2019+.

### 2. Entorno virtual (recomendado)

```bash
git clone https://github.com/aguitech/reconocimiento-facial-python.git
cd reconocimiento-facial-python
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Core
pip install -e .

# Backend dlib + face_recognition (incluye CMake + build dlib)
pip install dlib face_recognition

# Backend DeepFace (atributos)
pip install deepface

# Backend MediaPipe (face mesh, liveness)
pip install mediapipe

# Backend insightface (opcional, ArcFace)
pip install insightface onnxruntime
```

### 3. Verificar instalación

```python
import face_recognition
import cv2
import mediapipe as mp

print("face_recognition:", face_recognition.__version__)
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__)
```

## Uso

### API unificada (recomendado)

```python
from facekit import FaceEngine

# Carga el backend que quieras (face_recognition por default)
engine = FaceEngine(backend="face_recognition")

# Detección + landmarks
result = engine.detect("foto.jpg")
for face in result.faces:
    print(face.bbox)            # (x1, y1, x2, y2)
    print(face.landmarks)       # dict con eyes, nose, mouth, ...
    print(face.embedding)       # np.ndarray de 128 floats

# Verificación 1:1 (¿es la misma persona?)
match = engine.verify("persona_a.jpg", "persona_b.jpg")
print(match.distance, match.match)  # distancia < umbral → True

# Identificación 1:N
ids = engine.identify("desconocido.jpg", database=["ana.jpg", "luis.jpg", "maria.jpg"])
print(ids)  # [("ana.jpg", 0.42), ("luis.jpg", 0.55), ...]
```

### Scripts rápidos

```bash
# Detectar todas las caras en una imagen y dibujar bboxes
python3 scripts/detect_image.py --rect 0.55 foto.jpg -o salida.jpg

# Identificar a la persona frente a la webcam
python3 scripts/webcam_identify.py --db ./known_faces/

# Comparar 2 imágenes
python3 scripts/compare.py foto_a.jpg foto_b.jpg

# Análisis de atributos (edad / género / emoción)
python3 scripts/analyze_attributes.py foto.jpg
```

## Ejemplos

### 1. Detector con webcam (10 líneas)

```python
import cv2
from facekit import FaceEngine

engine = FaceEngine(backend="opencv")
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    for face in engine.detect(frame).faces:
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

### 2. Indexar una base de datos

```python
from pathlib import Path
from facekit import FaceEngine
from facekit.storage import EmbeddingDB

engine = FaceEngine(backend="face_recognition")
db = EmbeddingDB("./embeddings.h5")

for img_path in Path("./people").glob("*.jpg"):
    name = img_path.stem
    result = engine.detect(img_path)
    for face in result.faces:
        db.add(name=name, embedding=face.embedding, source=str(img_path))

db.save()
print(f"Indexed {len(db)} faces")
```

### 3. Análisis de atributos

```python
from deepface import DeepFace

objs = DeepFace.analyze(
    img_path="foto.jpg",
    actions=["age", "gender", "emotion", "race"],
    enforce_detection=True,
)

for face in objs:
    print({
        "age":     face["age"],
        "gender":  face["dominant_gender"],
        "emotion": face["dominant_emotion"],
        "race":    face["dominant_race"],
    })
```

## API

### `FaceEngine`

| Método | Descripción |
|--------|-------------|
| `detect(image, **kwargs)` | Detecta caras, devuelve `DetectionResult` con lista de `Face` |
| `verify(img1, img2, threshold=0.5)` | Verificación 1:1 — devuelve `VerifyResult` |
| `identify(unknown, database, top_k=3)` | Identificación 1:N — devuelve lista ordenada por similitud |
| `embed(image)` | Devuelve solo embeddings sin dibujar / reportar |

### `Face`

```python
@dataclass
class Face:
    bbox:      tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float
    landmarks: dict[str, np.ndarray]
    embedding: np.ndarray | None           # 128-D / 512-D según backend
    age:       int | None
    gender:    str | None
    emotion:   str | None
```

### `EmbeddingDB`

```python
db = EmbeddingDB(path)
db.add(name, embedding, source=None)
db.search(embedding, top_k=5)
db.save()
db.load()
```

## Estructura

```
reconocimiento-facial-python/
├── facekit/
│   ├── __init__.py
│   ├── engine.py             # FaceEngine unificado
│   ├── backends/
│   │   ├── opencv.py
│   │   ├── face_recognition.py
│   │   ├── mediapipe.py
│   │   ├── deepface.py
│   │   └── insightface.py
│   ├── storage.py            # EmbeddingDB
│   ├── utils.py              # I/O, resize, draw
│   └── types.py              # dataclasses: Face, DetectionResult, ...
├── scripts/
│   ├── detect_image.py
│   ├── webcam_identify.py
│   ├── compare.py
│   └── analyze_attributes.py
├── tests/
│   ├── test_engine.py
│   └── fixtures/
├── examples/
│   └── notebook.ipynb
├── pyproject.toml
├── README.md
└── LICENSE
```

## Performance

| Backend | Detección (FPS @ 1080p) | Embedding (ms) | Tamaño modelo |
|---------|--------------------------|----------------|---------------|
| OpenCV Haar | 60+ | — | 1 MB |
| OpenCV DNN YuNet | 50+ | — | 5 MB |
| MediaPipe | 80+ | — | 4 MB |
| face_recognition (HOG) | 12 | 200 | — |
| face_recognition (CNN) | 8 | 280 | 230 MB |
| DeepFace (Facenet) | 6 | 320 | 95 MB |
| insightface (ArcFace) | 25 | 35 (GPU) | 250 MB |

Mediciones en CPU Intel i7-1165G7 / GPU RTX 3060.

## Consideraciones éticas

- 🔒 **Consentimiento**: este software identifica personas. Úsalo solo con consentimiento explícito.
- ⚖️ **Privacidad**: el GDPR y otras leyes regulan el procesamiento biométrico. Asegúrate de cumplir.
- 🚫 **Anti-spoofing**: este repo NO hace liveness detection por default — una foto impresa pasa la verificación. Para producción real usa liveness (MediaPipe + depth / IR).
- 📊 **Sesgo**: los modelos tienen sesgos por edad / etnia / género. Mide tu caso de uso real antes de desplegar.

## License

MIT — úsalo, modifícalo, repártelo. Si te late, menciónanos.

---

<p align="center">
  Hecho con 🇨 por <a href="https://aguitech.com"><strong>AGUITECH</strong></a> ·
  Ingeniería + Diseño + Sistemas
</p>