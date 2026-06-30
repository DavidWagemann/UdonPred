"""CAID4-specific entry points for UdonPred.

These modules adapt UdonPred to the CAID4 submission rules: the protein
language model (ProstT5) is *not* shipped or run here. Instead, per-residue
embeddings are precomputed externally and passed in as ``.npy`` or ``.h5``
files; this package loads them, aligns them to the input sequences, and runs
the shared ONNX prediction heads from :mod:`udonpred`.
"""
