import pytest
from pyannote.core import Annotation, Segment

from diarization.rttm import read_audacity, read_rttm, write_audacity, write_rttm


def intervals(annotation):
    return [(segment.start, segment.end, label) for segment, _, label in annotation.itertracks(yield_label=True)]


def test_round_trip_with_overlap_and_coincident_segments(tmp_path):
    original = Annotation(uri="ev01")
    original[Segment(0.1234, 1.5678), "a"] = "ENTREVISTADOR"
    original[Segment(0.5, 2.3456), "b"] = "ENTREVISTADO"
    original[Segment(0.5, 2.3456), "c"] = "HABLANTE_3"
    rttm = tmp_path / "ev01.rttm"
    labels = tmp_path / "ev01.txt"
    write_rttm(original, rttm)
    recovered = read_rttm(rttm)
    write_audacity(recovered, labels)
    restored = read_audacity(labels, uri="ev01")
    assert restored.uri == original.uri
    expected = sorted(intervals(original))
    actual = sorted(intervals(restored))
    assert len(actual) == len(expected) == 3
    for (start, end, label), (want_start, want_end, want_label) in zip(actual, expected):
        assert start == pytest.approx(want_start, abs=0.001)
        assert end == pytest.approx(want_end, abs=0.001)
        assert label == want_label


def test_rejects_multiple_uris(tmp_path):
    path = tmp_path / "invalid.rttm"
    path.write_text(
        "SPEAKER ev01 1 0 1 <NA> <NA> A <NA> <NA>\n"
        "SPEAKER ev02 1 0 1 <NA> <NA> B <NA> <NA>\n"
    )
    with pytest.raises(ValueError, match="URI"):
        read_rttm(path)


def test_empty_rttm(tmp_path):
    path = tmp_path / "ev01.rttm"
    path.write_text("")
    annotation = read_rttm(path)
    assert annotation.uri == "ev01"
    assert list(annotation.itertracks()) == []
