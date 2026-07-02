"""Standalone offline tooling for UdonPred.

These are build-time / operator utilities that are not part of the shipped
inference API: ``embed`` generates ProstT5 embeddings (needs
``udonpred[embedding]``) and ``export`` converts trained checkpoints to ONNX
prediction heads (needs ``udonpred[training]``).
"""
