"""UdonPred: protein disorder prediction from ProstT5 embeddings.

Package layout:

* :mod:`udonpred.fasta`, :mod:`udonpred.inference` -- the lean, torch-free core
  (FASTA handling, ONNX prediction heads, smoothing). Depends only on the light
  base dependencies.
* :mod:`udonpred.heads`, :mod:`udonpred.datasets` -- resolve prediction heads,
  datasets, and embeddings from a local path or the Hugging Face Hub.
* :mod:`udonpred.output`, :mod:`udonpred.cli` -- post-processing, ``.caid``
  writing, and the prediction CLI flags.
* :mod:`udonpred.embedding` -- the ProstT5 backbone and the on-the-fly
  predictor. Needs the ``udonpred[embedding]`` extra (torch, transformers).
* :mod:`udonpred.training` -- model training / optimization. Needs the
  ``udonpred[training]`` extra.
* :mod:`udonpred.utils` -- offline tooling (``embed``, ``export``, ``publish``).

This module deliberately imports nothing: ``import udonpred`` stays torch-free
so the lean inference path never drags in the heavy PLM/training stack.
"""
