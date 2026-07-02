"""UdonPred: protein disorder prediction from ProstT5 embeddings.

Package layout:

* :mod:`udonpred.fasta`, :mod:`udonpred.inference` -- the lean, torch-free core
  (FASTA handling, ONNX prediction heads, smoothing). Depends only on the light
  base dependencies.
* :mod:`udonpred.caid` -- the CAID4 runner that consumes precomputed embeddings
  (also torch-free; this is what the inference container ships).
* :mod:`udonpred.embedding` -- the ProstT5 backbone and the on-the-fly
  predictor. Needs the ``udonpred[embedding]`` extra (torch, transformers).
* :mod:`udonpred.training` -- model training / optimization. Needs the
  ``udonpred[training]`` extra.
* :mod:`udonpred.utils` -- offline tooling (``embed``, ``export``).

This module deliberately imports nothing: ``import udonpred`` stays torch-free
so the lean inference path never drags in the heavy PLM/training stack.
"""
