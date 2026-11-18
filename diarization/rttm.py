"""Conversión entre anotaciones, RTTM y etiquetas de Audacity."""

from pathlib import Path

from pyannote.core import Annotation, Segment


def read_rttm(path: Path) -> Annotation:
    annotation = None
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 10 or fields[0] != "SPEAKER":
                raise ValueError(f"RTTM inválido en línea {line_number}")
            _, uri, _, start, duration, _, _, speaker, _, _ = fields
            if annotation is None:
                annotation = Annotation(uri=uri)
            elif annotation.uri != uri:
                raise ValueError("El RTTM contiene más de un URI")
            start, duration = float(start), float(duration)
            if start < 0 or duration <= 0:
                raise ValueError(f"Intervalo inválido en línea {line_number}")
            annotation[Segment(start, start + duration), line_number] = speaker
    return annotation if annotation is not None else Annotation(uri=Path(path).stem)


def write_rttm(annotation: Annotation, path: Path) -> None:
    if not annotation.uri or any(c.isspace() for c in annotation.uri):
        raise ValueError("La anotación necesita un URI sin espacios")
    with Path(path).open("w", encoding="utf-8") as target:
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            if not isinstance(speaker, str) or not speaker or any(c.isspace() for c in speaker):
                raise ValueError("Las etiquetas RTTM deben ser texto sin espacios")
            target.write(
                f"SPEAKER {annotation.uri} 1 {segment.start:.6f} "
                f"{segment.duration:.6f} <NA> <NA> {speaker} <NA> <NA>\n"
            )


def read_audacity(path: Path, *, uri: str) -> Annotation:
    annotation = Annotation(uri=uri)
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 3 or not fields[2]:
                raise ValueError(f"Etiqueta inválida en línea {line_number}")
            start, end = float(fields[0]), float(fields[1])
            if start < 0 or end <= start:
                raise ValueError(f"Intervalo inválido en línea {line_number}")
            annotation[Segment(start, end), line_number] = fields[2]
    return annotation


def write_audacity(annotation: Annotation, path: Path) -> None:
    with Path(path).open("w", encoding="utf-8") as target:
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            if not isinstance(speaker, str) or not speaker or any(c in speaker for c in "\t\r\n"):
                raise ValueError("Etiqueta de Audacity inválida")
            target.write(f"{segment.start:.6f}\t{segment.end:.6f}\t{speaker}\n")
