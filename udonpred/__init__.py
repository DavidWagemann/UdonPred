"""Shared UdonPred core: FASTA handling, ONNX head inference, and ProstT5 backbone.

These modules are backbone-agnostic where possible so that both the standard
``predict.py`` runner (which computes embeddings on the fly) and the CAID
runner (``caid/predict.py``, which consumes precomputed embeddings) can reuse
the same logic.
"""
