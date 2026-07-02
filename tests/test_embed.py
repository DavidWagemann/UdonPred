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


def test_embed_push_to_hub_calls_publish(tmp_path, monkeypatch):
    # generate() needs torch; instead test the push wiring in isolation.
    from udonpred.utils import embed

    called = {}

    def fake_publish(h5_path, target, split, plm, repo=None):
        called.update(target=target, split=split, plm=plm, repo=repo)

    monkeypatch.setattr(embed, "publish_embeddings", fake_publish)

    h5 = tmp_path / "e.h5"
    h5.write_bytes(b"x")
    embed.push_embeddings_to_hub(
        str(h5), dataset="trizod", split="test", backbone_name="Rostlab/ProstT5_fp16"
    )
    assert called == {"target": "trizod", "split": "test", "plm": "prostt5", "repo": None}
