import re

from udonpred.fasta import (
    format_predictions,
    read_fasta,
    safe_filename,
    sanitize_sequence,
)


def test_read_fasta_multi_entry(tmp_path):
    p = tmp_path / "in.fasta"
    p.write_text(">seq1\nMKT\nAY\n>seq2\nGGG\n")
    entries = read_fasta(str(p))
    assert entries == [("seq1", "MKTAY"), ("seq2", "GGG")]


def test_read_fasta_strips_internal_whitespace_and_blank_lines(tmp_path):
    p = tmp_path / "in.fasta"
    p.write_text(">seq1\nMK T\n\n A Y \n")
    entries = read_fasta(str(p))
    assert entries == [("seq1", "MKTAY")]


def test_sanitize_sequence_replaces_all_ambiguous_including_J():
    # B Z J U O and * must all map to X; standard residues and X untouched.
    assert sanitize_sequence("BZJUO*") == "XXXXXX"
    assert sanitize_sequence("MKXTAY") == "MKXTAY"


def test_format_predictions_emits_caid_rows():
    lines = format_predictions("hdr", "MK", [0.1, 0.92])
    assert lines[0] == ">hdr\n"
    assert lines[1] == "1\tM\t0.100\t\n"
    assert lines[2] == "2\tK\t0.920\t\n"


def test_format_predictions_emits_binary_column():
    lines = format_predictions("hdr", "MK", [0.1, 0.92], [0, 1])
    assert lines[1] == "1\tM\t0.100\t0\n"
    assert lines[2] == "2\tK\t0.920\t1\n"


def test_format_predictions_unwraps_trailing_axis_in_both_columns():
    # heads emit (L, 1); binarize_scores preserves that shape
    lines = format_predictions("hdr", "M", [[0.42]], [[1]])
    assert lines[1] == "1\tM\t0.420\t1\n"


def test_safe_filename_replaces_path_unsafe_characters():
    assert safe_filename("sp|P04637|P53_HUMAN") == "sp_P04637_P53_HUMAN"
    assert safe_filename("P04637 cellular tumor antigen") == "P04637_cellular_tumor_antigen"
    assert safe_filename("a/b") == "a_b"


def test_format_predictions_always_uses_three_decimals():
    scores = [0.0, 1.0, 0.5, 1e-9, 0.12349, 0.9999, -0.4, 123.456789, 34.0]
    lines = format_predictions("hdr", "M" * len(scores), scores)
    for line in lines[1:]:
        score = line.split("\t")[2]
        assert re.fullmatch(r"-?\d+\.\d{3}", score), score


def test_format_predictions_rounds_rather_than_truncates():
    lines = format_predictions("hdr", "MK", [0.12349, 0.9999])
    assert lines[1].split("\t")[2] == "0.123"
    assert lines[2].split("\t")[2] == "1.000"
