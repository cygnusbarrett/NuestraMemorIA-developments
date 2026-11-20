"""Compara diarizaciones sin tratar ninguna como referencia humana."""

import argparse
import json
from pathlib import Path

from pyannote.metrics.diarization import DiarizationErrorRate

from diarization.rttm import read_rttm

RESULTS_ROOT = Path("eval/results")
DEFAULT_BACKEND = "pyannote31"


def compare_annotations(a, b, max_intervals=5):
    metric = DiarizationErrorRate(collar=0, skip_overlap=False)
    score = metric(a, b)
    mapping = metric.optimal_mapping(a, b)
    tracks_a = list(a.itertracks(yield_label=True))
    tracks_b = list(b.itertracks(yield_label=True))
    boundaries = sorted({t for segment, _, _ in tracks_a + tracks_b for t in (segment.start, segment.end)})
    intervals = []
    for start, end in zip(boundaries, boundaries[1:]):
        left = sorted({label for segment, _, label in tracks_a if segment.start < end and segment.end > start})
        right = sorted({label for segment, _, label in tracks_b if segment.start < end and segment.end > start})
        if set(left) == {mapping.get(label, ("sin correspondencia", label)) for label in right}:
            continue
        if intervals and intervals[-1]['fin'] == start and intervals[-1]['a'] == left and intervals[-1]['b'] == right:
            intervals[-1]['fin'] = end
        else:
            intervals.append({'inicio': start, 'fin': end, 'a': left, 'b': right})
    return {'discrepancia_der': score, 'tramos': sorted(intervals, key=lambda item: item['fin'] - item['inicio'], reverse=True)[:max_intervals]}


def compare(a, b):
    left = {p.stem: p for p in (a / 'hypotheses').glob('*.rttm')}
    right = {p.stem: p for p in (b / 'hypotheses').glob('*.rttm')}
    if not left or left.keys() != right.keys():
        raise ValueError('Las variantes deben tener los mismos fragmentos')
    return {uri: compare_annotations(read_rttm(left[uri]), read_rttm(right[uri])) for uri in sorted(left)}


def resolve_variant_path(identifier, *, root=RESULTS_ROOT):
    """Resuelve `<backend>/<variante>` o `<variante>` (pyannote31) bajo eval/results."""
    text = str(identifier).strip()
    if not text or text in (".", "..") or "\\" in text:
        raise ValueError("Identificador de variante inválido")
    parts = Path(text).parts
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("Identificador de variante inválido")
    if len(parts) == 1:
        relative = Path(DEFAULT_BACKEND) / parts[0]
    elif len(parts) == 2:
        relative = Path(parts[0]) / parts[1]
    else:
        raise ValueError("Identificador de variante inválido")
    root = Path(root).resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("La variante sale de eval/results") from error
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    args = parser.parse_args()
    try:
        left = resolve_variant_path(args.a)
        right = resolve_variant_path(args.b)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(compare(left, right), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
