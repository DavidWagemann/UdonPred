"""The on-the-fly embedding pipeline, end to end with a fake ProstT5.

The tokenizer and backbone are replaced by small deterministic stand-ins with
the Hugging Face call signatures, so tokenization, prefix trimming, batching,
and writing run for real. Needs ``torch`` (the ``embedding`` extra).
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import udonpred.embedding.predict as predict
import udonpred.utils.embed as embed

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
VOCAB = {"<AA2fold>": 3, **{aa: 4 + i for i, aa in enumerate(ALPHABET)}}
EOS = 2
ENTRIES = [("P04637", "MEEPQSDPSVEPPLSQETFS"), ("sp|P38398|X", "MDLSALRVEEV"), ("amb", "MKBZUO")]


class FakeTokenizer:
    def batch_encode_plus(self, texts, add_special_tokens=True, padding="longest",
                          return_tensors="pt"):
        ids = [[VOCAB[token] for token in text.split(" ")] + [EOS] for text in texts]
        width = max(map(len, ids))
        input_ids = torch.zeros(len(ids), width, dtype=torch.long)
        mask = torch.zeros(len(ids), width, dtype=torch.long)
        for row, seq in enumerate(ids):
            input_ids[row, : len(seq)] = torch.tensor(seq)
            mask[row, : len(seq)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


class FakeBackbone:
    """Per-token lookup plus a positional term, so padding can't leak in."""

    def __init__(self):
        gen = torch.Generator().manual_seed(0)
        self.table = torch.randn(32, 1024, generator=gen)
        self.pos = torch.randn(512, 1024, generator=gen)

    def __call__(self, input_ids, attention_mask=None):
        hidden = self.table[input_ids] + self.pos[: input_ids.shape[1]]
        return SimpleNamespace(last_hidden_state=hidden)

    def residue_rows(self, seq):
        """What the backbone emits for ``seq``'s residues (prefix trimmed)."""
        ids = torch.tensor([VOCAB[aa] for aa in seq])
        return (self.table[ids] + self.pos[1 : 1 + len(seq)]).numpy()


@pytest.fixture
def fake_prostt5(monkeypatch):
    backbone = FakeBackbone()
    for module in (predict, embed):
        monkeypatch.setattr(module, "resolve_device", lambda device: "cpu")
        monkeypatch.setattr(module, "load_tokenizer", lambda: FakeTokenizer())
        monkeypatch.setattr(module, "load_backbone", lambda device, dtype: backbone)
    return backbone


@pytest.fixture
def fasta(tmp_path):
    path = tmp_path / "in.fasta"
    path.write_text("".join(f">{h}\n{s}\n" for h, s in ENTRIES))
    return path


def test_embed_writes_one_trimmed_float32_array_per_header(tmp_path, fasta, fake_prostt5):
    h5py = pytest.importorskip("h5py")
    out = tmp_path / "emb.h5"
    embed.generate(str(fasta), str(out), "cpu", 2000)

    with h5py.File(out) as f:
        assert sorted(f) == sorted(h for h, _ in ENTRIES)
        for header, seq in ENTRIES:
            arr = f[header][:]
            assert arr.shape == (len(seq), 1024)
            assert arr.dtype == np.float32
            # ambiguous residues are embedded as X
            expected = fake_prostt5.residue_rows(seq.translate(str.maketrans("BZUO", "XXXX")))
            np.testing.assert_allclose(arr, expected, rtol=1e-6)


def test_embed_npy_holds_the_single_sequence(tmp_path, fake_prostt5):
    fasta = tmp_path / "one.fasta"
    fasta.write_text(">only\nMKTAY\n")
    embed.generate(str(fasta), str(tmp_path / "emb.npy"), "cpu", 2000)
    arr = np.load(tmp_path / "emb.npy")
    np.testing.assert_allclose(arr, fake_prostt5.residue_rows("MKTAY"), rtol=1e-6)


def test_embeddings_do_not_depend_on_batching(tmp_path, fasta, fake_prostt5):
    h5py = pytest.importorskip("h5py")
    embed.generate(str(fasta), str(tmp_path / "one_batch.h5"), "cpu", 2000)
    embed.generate(str(fasta), str(tmp_path / "per_seq.h5"), "cpu", 5)
    with h5py.File(tmp_path / "one_batch.h5") as a, h5py.File(tmp_path / "per_seq.h5") as b:
        for header, _ in ENTRIES:
            np.testing.assert_array_equal(a[header][:], b[header][:])


def _predict(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["udonpred-predict", *argv])
    predict.main()


def test_on_the_fly_predictions_do_not_depend_on_batching(
    tmp_path, fasta, weights_dir, fake_prostt5, monkeypatch
):
    for label, batch_size in (("one_batch", "2000"), ("per_seq", "5")):
        _predict(monkeypatch, str(fasta), str(weights_dir), "-t", "trizod", "chezod",
                 "-b", batch_size, "-o", str(tmp_path / label))

    for target in ("trizod", "chezod"):
        names = sorted(p.name for p in (tmp_path / "one_batch" / target).glob("*.caid"))
        assert names == ["P04637.caid", "amb.caid", "sp_P38398_X.caid"]
        for name in names:
            one = (tmp_path / "one_batch" / target / name).read_text()
            per_seq = (tmp_path / "per_seq" / target / name).read_text()
            assert one == per_seq, (target, name)
