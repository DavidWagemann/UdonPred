"""ProstT5 embedding backbone and the on-the-fly predictor.

Everything in this subpackage depends on the heavy PLM stack (torch,
transformers). It is imported only by the ``udonpred[embedding]`` workflows and
never by the lean, torch-free CAID inference path (:mod:`udonpred.caid`).
"""
