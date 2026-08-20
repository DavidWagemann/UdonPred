import numpy as np
import pytest

from udonpred.output import CaidWriter, postprocess_scores


def test_postprocess_returns_scores_and_binary_of_matching_shape():
    raw = np.array([[10.0], [2.0], [20.0]])
    scores, binary = postprocess_scores(raw, "chezod", smooth=0)
    assert scores.shape == binary.shape == raw.shape
    # chezod is order-positive: only the low raw value is disordered
    assert binary.ravel().tolist() == [0, 1, 0]
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_postprocess_binary_ignores_normalize_flag():
    raw = np.array([1.0, 5.0, 70.0, 100.0])
    _, with_norm = postprocess_scores(raw, "plddt", smooth=0, normalize=True)
    raw_scores, without_norm = postprocess_scores(
        raw, "plddt", smooth=0, normalize=False
    )
    assert with_norm.tolist() == without_norm.tolist()
    # --no-normalize leaves the score column on the head's own scale
    assert raw_scores.tolist() == raw.tolist()


def test_postprocess_thresholds_the_smoothed_scores():
    # a lone spike is smoothed below the cutoff, so it must not be called
    raw = np.zeros((21, 1))
    raw[10, 0] = 1.0
    _, binary = postprocess_scores(raw, "trizod", smooth=1.5)
    assert binary.sum() == 0
    _, unsmoothed = postprocess_scores(raw, "trizod", smooth=0)
    assert unsmoothed.sum() == 1


def test_writer_creates_one_dir_per_target_and_file_per_protein(tmp_path):
    writer = CaidWriter(str(tmp_path), ["trizod", "chezod"], smooth=0)
    for target in ("trizod", "chezod"):
        assert (tmp_path / target).is_dir()
    writer.write("trizod", "P04637", "MK", np.array([[0.9], [0.1]]))
    writer.write("chezod", "sp|P38398|X", "MK", np.array([[0.9], [0.1]]))

    out = (tmp_path / "trizod" / "P04637.caid").read_text()
    assert out == ">P04637\n1\tM\t0.900\t1\n2\tK\t0.100\t0\n"
    # header sanitised for the filename, preserved inside the file
    chezod = tmp_path / "chezod" / "sp_P38398_X.caid"
    assert chezod.read_text().startswith(">sp|P38398|X\n")


def test_writer_falls_back_to_stdout(capsys):
    writer = CaidWriter(None, ["trizod"], smooth=0)
    writer.write("trizod", "seq1", "M", np.array([[0.9]]))
    assert capsys.readouterr().out == ">seq1\n1\tM\t0.900\t1\n"


def test_writer_labels_targets_on_stdout_only_when_multiple(capsys):
    CaidWriter(None, ["trizod"], smooth=0).write("trizod", "s", "M", np.array([[0.9]]))
    assert "# target:" not in capsys.readouterr().out

    multi = CaidWriter(None, ["trizod", "chezod"], smooth=0)
    multi.write("trizod", "s", "M", np.array([[0.9]]))
    assert "# target: trizod\n" in capsys.readouterr().out


def test_writer_rejects_unregistered_target(tmp_path):
    writer = CaidWriter(str(tmp_path), ["bogus"], smooth=0)
    with pytest.raises(ValueError, match="No policy registered"):
        writer.write("bogus", "seq1", "M", np.array([[0.5]]))
