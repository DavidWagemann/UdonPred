import numpy as np

import udonpred.datasets as ds
from udonpred.caid import predict as caid


def test_run_resolves_hub_embeddings_reference(tmp_path, monkeypatch, weights_dir):
    # A fake .h5 the "Hub reference" resolves to.
    import h5py

    emb_path = tmp_path / "test.h5"
    with h5py.File(emb_path, "w") as f:
        f["seq1"] = np.random.randn(10, 1024).astype(np.float32)
    (tmp_path / "in.fasta").write_text(">seq1\nMKTAYIAKQR\n")

    seen = {}

    def fake_resolve(source, revision=None):
        seen["source"] = source
        return emb_path

    monkeypatch.setattr(caid, "resolve_embeddings_ref", fake_resolve)

    caid.run(
        fasta=str(tmp_path / "in.fasta"),
        model_dir=str(weights_dir),
        embeddings="udonpred/datasets:trizod/embeddings/prostt5/test.h5",
        target=["trizod"],
        output_path=str(tmp_path / "out"),
        device="cpu",
        threads=None,
        smooth=1.5,
    )

    assert seen["source"] == "udonpred/datasets:trizod/embeddings/prostt5/test.h5"
    out_file = tmp_path / "out" / "trizod" / "seq1.caid"
    assert out_file.read_text().startswith(">seq1\n")
