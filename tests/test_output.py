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


# --- timings.csv --------------------------------------------------------------


def _timings(path):
    lines = path.read_text().splitlines()
    return lines[0], lines[1], lines[2:]


def test_timings_csv_written_per_flavor_directory(tmp_path):
    with CaidWriter(str(tmp_path), ["trizod", "chezod"], smooth=0) as writer:
        for target in ("trizod", "chezod"):
            for header in ("P04637", "P38398"):
                with writer.timing(target, header):
                    writer.write(target, header, "MK", np.array([[0.9], [0.1]]))

    for target in ("trizod", "chezod"):
        banner, head, rows = _timings(tmp_path / target / "timings.csv")
        assert banner.startswith("# Running UdonPred, started ")
        assert head == "sequence,milliseconds"
        assert [r.split(",")[0] for r in rows] == ["P04637", "P38398"]
        for row in rows:
            # milliseconds must be a bare integer
            assert row.split(",")[1].isdigit()


def test_timings_rows_only_cover_that_flavor(tmp_path):
    with CaidWriter(str(tmp_path), ["trizod", "chezod"], smooth=0) as writer:
        with writer.timing("trizod", "only_trizod"):
            writer.write("trizod", "only_trizod", "M", np.array([[0.9]]))

    assert _timings(tmp_path / "trizod" / "timings.csv")[2] != []
    # the head that produced nothing still gets a well-formed, empty file
    banner, head, rows = _timings(tmp_path / "chezod" / "timings.csv")
    assert head == "sequence,milliseconds"
    assert rows == []


def test_timings_includes_attributed_shared_cost(tmp_path):
    with CaidWriter(str(tmp_path), ["trizod"], smooth=0) as writer:
        with writer.timing("trizod", "P04637", extra_ms=500.0):
            writer.write("trizod", "P04637", "M", np.array([[0.9]]))

    row = _timings(tmp_path / "trizod" / "timings.csv")[2][0]
    assert int(row.split(",")[1]) >= 500


def test_timings_recorded_even_if_the_work_raises(tmp_path):
    with CaidWriter(str(tmp_path), ["trizod"], smooth=0) as writer:
        with pytest.raises(RuntimeError):
            with writer.timing("trizod", "P04637"):
                raise RuntimeError("boom")

    assert _timings(tmp_path / "trizod" / "timings.csv")[2][0].startswith("P04637,")


def test_timings_quotes_headers_containing_commas(tmp_path):
    with CaidWriter(str(tmp_path), ["trizod"], smooth=0) as writer:
        with writer.timing("trizod", "P04637, isoform 2"):
            writer.write("trizod", "P04637, isoform 2", "M", np.array([[0.9]]))

    import csv

    with (tmp_path / "trizod" / "timings.csv").open() as handle:
        next(handle)  # banner
        rows = list(csv.reader(handle))
    assert rows[0] == ["sequence", "milliseconds"]
    assert rows[1][0] == "P04637, isoform 2"


def test_no_timings_file_in_stdout_mode(tmp_path, capsys):
    with CaidWriter(None, ["trizod"], smooth=0) as writer:
        with writer.timing("trizod", "P04637"):
            writer.write("trizod", "P04637", "M", np.array([[0.9]]))
    capsys.readouterr()
    assert not list(tmp_path.iterdir())


def test_format_started_matches_the_caid_banner_shape():
    import re
    import time

    from udonpred.output import format_started

    # "Sun Feb  5 10:20:57 CET 2023" — day space-padded to width 2
    stamp = format_started(time.time())
    assert re.fullmatch(r"[A-Z][a-z]{2} [A-Z][a-z]{2} [ \d]\d \d{2}:\d{2}:\d{2} .+ \d{4}", stamp), stamp

    single_digit_day = time.mktime((2023, 2, 5, 10, 20, 57, 0, 0, -1))
    assert format_started(single_digit_day).startswith("Sun Feb  5 10:20:57")
