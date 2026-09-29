from udonpred.fasta import read_fasta, sanitize_sequence


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
