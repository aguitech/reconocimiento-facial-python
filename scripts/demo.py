"""Demo end-to-end: genera imágenes, registra personas, reconoce.

Uso:
    python scripts/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permitir ejecutar como `python scripts/demo.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reconocimiento_facial.config import get_settings, reload_settings  # noqa: E402
from reconocimiento_facial.pipeline import FacePipeline  # noqa: E402
from reconocimiento_facial.utils import configure_logging  # noqa: E402

from generate_synthetic_faces import main as generate_synthetic  # noqa: E402


def run_demo() -> None:
    settings = get_settings()
    # Usar un DB separado para la demo.
    settings.db_path = settings.data_dir / "demo.db"
    configure_logging("default")

    # 1) Generar imágenes sintéticas.
    print("=" * 60)
    print("PASO 1 — Generar imágenes sintéticas")
    print("=" * 60)
    image_dir = generate_synthetic(n_people=3, n_per_person=3)
    images = sorted(image_dir.glob("*.png"))

    # 2) Registrar personas.
    print("\n" + "=" * 60)
    print("PASO 2 — Registrar personas (con imágenes sintéticas)")
    print("=" * 60)
    print("⚠️  Nota: las imágenes sintéticas NO son rostros reales,")
    print("   dlib no las detectará. Usa fotos reales para resultados reales.")

    with FacePipeline() as pipeline:
        # Solo registrar las primeras (mejor caso).
        registered = 0
        person_images = {}
        for img_path in images:
            person_name = img_path.stem.rsplit("_", 1)[0]
            person_images.setdefault(person_name, []).append(img_path)

        for name, paths in person_images.items():
            try:
                pipeline.register_person(name=name, image_paths=paths)
                registered += 1
            except ValueError as exc:
                print(f"  [advertencia] {name}: {exc}")

        print(f"\n{registered} personas registradas con éxito")

        # 3) Reconocer.
        print("\n" + "=" * 60)
        print("PASO 3 — Reconocer en imágenes")
        print("=" * 60)
        for img_path in images[:3]:
            try:
                r = pipeline.recognize_image(img_path)
                print(f"\n{img_path.name}: {len(r.faces)} rostro(s) en {r.elapsed_ms:.0f} ms")
                for match in r.faces:
                    label = match.person.name if match.matched else "Desconocido"
                    print(f"   → {label}")
            except Exception as exc:
                print(f"  [error] {img_path.name}: {exc}")

    print("\n" + "=" * 60)
    print("Demo terminada. Para un demo con rostros REALES, sustituye")
    print("las imágenes sintéticas por fotos tuyas en data/known_faces/.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()