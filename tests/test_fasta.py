from udonpred.fasta import read_fasta, sanitize_sequence, format_predictions


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
