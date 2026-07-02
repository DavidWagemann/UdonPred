"""Unit tests for :mod:`udonpred.utils.embed`.

``embed.generate`` validates its inputs before touching ``torch`` (its import is
deferred), so the argument-validation branches below run torch-free.
"""

import pytest

from udonpred.utils.embed import generate


def _fasta(tmp_path, entries):
    path = tmp_path / "in.fasta"
    path.write_text("".join(f">{h}\n{s}\n" for h, s in entries))
    return str(path)


def test_generate_rejects_empty_fasta(tmp_path):
    fasta = _fasta(tmp_path, [])
    with pytest.raises(ValueError, match="No FASTA entries"):
        generate(fasta, str(tmp_path / "out.h5"), "cpu", 2000)


def test_generate_rejects_unknown_suffix(tmp_path):
    fasta = _fasta(tmp_path, [("seq1", "MKTAYIAK")])
    with pytest.raises(ValueError, match="Output must end"):
        generate(fasta, str(tmp_path / "out.txt"), "cpu", 2000)


def test_generate_rejects_npy_for_multiple_sequences(tmp_path):
    fasta = _fasta(tmp_path, [("seq1", "MKTAYIAK"), ("seq2", "GGGCCC")])
    with pytest.raises(ValueError, match="single sequence"):
        generate(fasta, str(tmp_path / "out.npy"), "cpu", 2000)
