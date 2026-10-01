"""Genera imágenes sintéticas con caras para probar el sistema sin datos reales.

Crea imágenes en `data/synthetic_faces/` con rostros estilo emoji de distintos colores
(no son caras reales — para una demo end-to-end, sustitúyelas por fotos reales).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from reconocimiento_facial.config import get_settings


def synth_face(name: str, hue: int, size: int = 200) -> Image.Image:
    """Genera una 'cara' muy simple: óvalo amarillo con ojos y boca.

    El objetivo NO es realismo sino tener imágenes con un patrón consistente
    por persona para validar el pipeline end-to-end.
    """
    img = Image.new("RGB", (size, size), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Cara (óvalo).
    draw.ellipse((20, 20, size - 20, size - 20), fill=(hue, 180, 100))
    # Ojos.
    eye_y = size // 2 - 20
    draw.ellipse((size // 3 - 8, eye_y - 8, size // 3 + 8, eye_y + 8), fill=(0, 0, 0))
    draw.ellipse((2 * size // 3 - 8, eye_y - 8, 2 * size // 3 + 8, eye_y + 8), fill=(0, 0, 0))
    # Boca.
    draw.arc(
        (size // 3, 2 * size // 3, 2 * size // 3, size - 30),
        start=0,
        end=180,
        fill=(0, 0, 0),
        width=3,
    )
    draw.text((10, 10), name, fill=(0, 0, 0))
    return img


def main(n_people: int = 3, n_per_person: int = 3) -> Path:
    """Genera n_people × n_per_person imágenes sintéticas."""
    settings = get_settings()
    out_dir = settings.data_dir / "synthetic_faces"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    created = 0
    for person_idx in range(n_people):
        name = f"synth_{person_idx + 1}"
        hue = int(rng.integers(0, 255))
        for sample_idx in range(n_per_person):
            # Variar brillo/contraste para tener muestras distintas.
            shift = int(rng.integers(-20, 20))
            img = synth_face(name=name, hue=hue)
            arr = np.array(img).astype(np.int16)
            arr = np.clip(arr + shift, 0, 255).astype(np.uint8)
            out_path = out_dir / f"{name}_{sample_idx + 1}.png"
            Image.fromarray(arr).save(out_path)
            created += 1

    print(f"✔ {created} imágenes sintéticas creadas en {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera imágenes de prueba sintéticas.")
    parser.add_argument("--n-people", type=int, default=3)
    parser.add_argument("--n-per-person", type=int, default=3)
    args = parser.parse_args()
    main(args.n_people, args.n_per_person)