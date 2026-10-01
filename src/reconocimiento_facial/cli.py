"""Interfaz de línea de comandos (Typer).

Comandos:
    rf-detect register      Registra una persona desde imágenes.
    rf-detect recognize     Reconoce rostros en una imagen estática.
    rf-detect webcam        Reconoce rostros en vivo desde la webcam.
    rf-detect list          Lista todas las personas registradas.
    rf-detect delete        Elimina una persona.
    rf-detect stats         Estadísticas del sistema.
    rf-detect serve         Levanta el servidor FastAPI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional as OptionalType  # noqa: F811  (sólo para rename)
from typing import Optional  # noqa: F401

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import get_settings
from .pipeline import FacePipeline
from .recognizer import FaceRecognizer
from .utils import configure_logging, logger

app = typer.Typer(
    name="rf-detect",
    help="Sistema de reconocimiento facial en Python.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold green]rf-detect[/bold green] v{__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(  # noqa: D401
    version: bool = typer.Option(False, "--version", "-V", callback=_version_callback),
) -> None:
    """rf-detect — reconocimiento facial CLI."""


# --- register ---------------------------------------------------------------

register_app = typer.Typer(help="Registra personas en la base de datos.")
app.add_typer(register_app, name="register")


@register_app.command("person")
def register_person(
    name: str = typer.Option(..., "--name", "-n", help="Nombre único de la persona."),
    images: list[Path] = typer.Option(  # noqa: B008
        ...,
        "--image",
        "-i",
        help="Ruta(s) a imágenes. Se puede repetir.",
    ),
    notes: str = typer.Option("", "--notes", help="Notas opcionales."),
    jitters: int = typer.Option(
        1, "--jitters", "-j", min=0, max=100, help="Veces que se muestrea el rostro."
    ),
) -> None:
    """Registra una persona a partir de imágenes."""
    settings = get_settings()
    configure_logging(settings.log_level)

    if not images:
        console.print("[red]Debes pasar al menos una imagen con --image.[/red]")
        raise typer.Exit(code=1)

    with FacePipeline() as pipeline:
        try:
            person = pipeline.register_person(name=name, image_paths=images, notes=notes, num_jitters=jitters)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        console.print(
            f"[green]✔ Persona '{person.name}' registrada con "
            f"{pipeline.database.count_samples(person.id)} muestra(s)[/green]"
        )


# --- recognize --------------------------------------------------------------

recognize_app = typer.Typer(help="Reconoce rostros en imágenes o webcam.")
app.add_typer(recognize_app, name="recognize")


@recognize_app.command("image")
def recognize_image(
    image_path: Path = typer.Argument(..., help="Ruta a la imagen a analizar."),
    output: Optional[Path] = typer.Option(  # noqa: UP007
        None, "--output", "-o", help="Guarda la imagen anotada en esta ruta."
    ),
    show: bool = typer.Option(False, "--show", "-s", help="Muestra la imagen con cv2."),
) -> None:
    """Reconoce rostros en una imagen estática."""
    settings = get_settings()
    configure_logging(settings.log_level)

    if not image_path.exists():
        console.print(f"[red]No existe la imagen:[/red] {image_path}")
        raise typer.Exit(code=1)

    from .utils import load_image

    with FacePipeline() as pipeline:
        result = pipeline.recognize_image(image_path)
        image = load_image(image_path, max_size=settings.max_image_size)

        if result.faces:
            table = Table(title=f"Rostros en {image_path.name}")
            table.add_column("#", justify="right")
            table.add_column("Persona")
            table.add_column("Distancia", justify="right")
            table.add_column("Confianza", justify="right")
            for idx, match in enumerate(result.faces, start=1):
                name = match.person.name if match.matched else "[red]Desconocido[/red]"
                table.add_row(
                    str(idx),
                    name,
                    f"{match.distance:.4f}" if not (match.distance != match.distance) else "—",
                    f"{match.confidence:.1%}",
                )
            console.print(table)
        else:
            console.print("[yellow]No se detectaron rostros.[/yellow]")

        console.print(f"[dim]Tiempo: {result.elapsed_ms:.1f} ms[/dim]")

        if output or show:
            annotated = pipeline.draw_results(image, result)
            if output:
                import cv2

                cv2.imwrite(str(output), annotated)
                console.print(f"[green]✔ Imagen guardada en {output}[/green]")
            if show:
                import cv2

                cv2.imshow("rf-detect", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


@recognize_app.command("webcam")
def recognize_webcam(
    camera: int = typer.Option(0, "--camera", "-c", help="Índice de la cámara."),
    skip: int = typer.Option(2, "--skip", "-k", min=1, help="Procesar 1 de cada N frames."),
    tolerance: Optional[float] = typer.Option(  # noqa: UP007
        None, "--tolerance", "-t", help="Sobrescribe la tolerancia de coincidencia."
    ),
) -> None:
    """Reconoce rostros en vivo desde la webcam."""
    settings = get_settings()
    configure_logging(settings.log_level)

    with FacePipeline() as pipeline:
        if tolerance is not None:
            pipeline.recognizer.tolerance = tolerance  # type: ignore[assignment]
            logger.info("Tolerancia ajustada a {}", tolerance)

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            console.print("[red]opencv-python es requerido.[/red]")
            raise typer.Exit(code=1) from exc

        console.print(f"[bold]Webcam {camera} — presiona Q para salir[/bold]")
        try:
            for result in pipeline.iter_webcam(camera_index=camera, skip_frames=skip):
                # Necesitamos el frame original para dibujar; reabrimos la cámara
                # o usamos el resultado parcial. Por rendimiento, simplificamos aquí
                # dibujando solo el conteo.
                faces_count = len(result.faces)
                matched = sum(1 for m in result.faces if m.matched)
                console.print(
                    f"[dim]frame {result.image_path} — "
                    f"{faces_count} rostro(s), {matched} reconocido(s), "
                    f"{result.elapsed_ms:.0f} ms[/dim]"
                )
                for m in result.faces:
                    label = m.person.name if m.matched else "Desconocido"
                    console.print(f"   → {label} ({m.confidence:.0%})")
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrumpido por el usuario.[/yellow]")


# --- list / delete / stats ---------------------------------------------------

@app.command("list")
def list_persons() -> None:
    """Lista todas las personas registradas."""
    settings = get_settings()
    configure_logging(settings.log_level)

    with FacePipeline() as pipeline:
        persons = pipeline.database.list_persons()
        if not persons:
            console.print("[yellow]No hay personas registradas.[/yellow]")
            return

        table = Table(title=f"Personas registradas ({len(persons)})")
        table.add_column("ID", style="dim")
        table.add_column("Nombre", style="bold")
        table.add_column("Muestras", justify="right")
        table.add_column("Creado")
        for p in persons:
            table.add_row(
                p.id[:8] + "…",
                p.name,
                str(p.sample_count),
                p.created_at.strftime("%Y-%m-%d %H:%M"),
            )
        console.print(table)


@app.command("delete")
def delete_person_cmd(
    person_id: str = typer.Argument(..., help="ID (o prefijo) de la persona a eliminar."),
    force: bool = typer.Option(False, "--yes", "-y", help="No pedir confirmación."),
) -> None:
    """Elimina una persona."""
    settings = get_settings()
    configure_logging(settings.log_level)

    with FacePipeline() as pipeline:
        # Permitir borrar por prefijo del ID.
        target = None
        for p in pipeline.database.list_persons():
            if p.id == person_id or p.id.startswith(person_id):
                target = p
                break
        if target is None:
            console.print(f"[red]No se encontró persona con id {person_id}[/red]")
            raise typer.Exit(code=1)

        if not force:
            confirm = typer.confirm(f"¿Eliminar '{target.name}' y sus {target.sample_count} muestra(s)?")
            if not confirm:
                console.print("[yellow]Cancelado.[/yellow]")
                return

        if pipeline.database.delete_person(target.id):
            pipeline.recognizer.mark_dirty()
            console.print(f"[green]✔ Persona '{target.name}' eliminada.[/green]")


@app.command("stats")
def stats() -> None:
    """Muestra estadísticas del sistema."""
    settings = get_settings()
    configure_logging(settings.log_level)

    with FacePipeline() as pipeline:
        persons = pipeline.database.list_persons()
        total_samples = pipeline.database.count_samples()
        console.print(
            f"[bold]rf-detect v{__version__}[/bold]\n"
            f"  Base de datos: {pipeline.database.db_path}\n"
            f"  Personas: {len(persons)}\n"
            f"  Muestras totales: {total_samples}\n"
            f"  Tolerancia: {pipeline.recognizer.tolerance}\n"
            f"  Detector: {settings.detector_backend}\n"
            f"  Métrica: {settings.distance_metric}"
        )


@app.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),  # noqa: S104
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", help="Hot-reload (desarrollo)."),
) -> None:
    """Levanta el servidor FastAPI."""
    import uvicorn

    console.print(f"[bold green]Sirviendo en http://{host}:{port}[/bold green]")
    uvicorn.run("reconocimiento_facial.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()  # pragma: no cover