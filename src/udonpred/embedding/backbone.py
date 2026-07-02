"""ProstT5 backbone loading, tokenization, and on-the-fly embedding.

This is the *online* embedding path used by the on-the-fly runner
(:mod:`udonpred.embedding.predict`) and the ``udonpred-embed`` helper
(:mod:`udonpred.utils.embed`). It is deliberately isolated from the rest of
:mod:`udonpred` so that the CAID runner (:mod:`udonpred.caid.predict`) can
import the FASTA and ONNX-head helpers without pulling in
``torch``/``transformers`` or the protein language model itself.

Importing this module requires ``torch`` and ``transformers``; the CAID
container does not install them.
"""

from importlib import import_module
from typing import List, Tuple

from ..fasta import sanitize_sequence

BACKBONE_NAME = "Rostlab/ProstT5_fp16"
BACKBONE_MODEL_TYPE = "T5EncoderModel"
BACKBONE_TOKENIZER_TYPE = "T5Tokenizer"
PREFIX_TOKEN = "<AA2fold>"
# Number of leading prefix tokens to strip from the per-token embedding.
PREFIX_TOKEN_LEN = 1


def resolve_device(device: str) -> str:
    """Resolve ``auto``/``cpu``/``cuda`` to a concrete device string."""
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "cuda"
    if device == "cpu":
        return "cpu"
    raise ValueError("Device must be one of: auto, cpu, cuda")


def load_tokenizer():
    """Load the ProstT5 tokenizer."""
    tokenizer_base = getattr(import_module("transformers"), BACKBONE_TOKENIZER_TYPE)
    return tokenizer_base.from_pretrained(
        BACKBONE_NAME,
        use_fast=False,
        do_lower_case=False,
        legacy=True,
    )


def load_backbone(torch_device: str, dtype):
    """Load the ProstT5 encoder onto ``torch_device`` with the given dtype."""
    backbone_base = getattr(import_module("transformers"), BACKBONE_MODEL_TYPE)
    model = backbone_base.from_pretrained(BACKBONE_NAME)
    model.config.output_attentions = False
    model.config.output_hidden_states = False
    return model.eval().to(torch_device).to(dtype)


def tokenize_batch(tokenizer, seqs: List[str], torch_device: str):
    """Tokenize a batch of sequences with the ``<AA2fold>`` prefix.

    Ambiguous/non-standard residues (B, Z, J, U, O, ``*``) are replaced with
    ``X`` via :func:`udonpred.fasta.sanitize_sequence` before tokenization.
    """
    texts = [
        f"{PREFIX_TOKEN} " + " ".join(list(sanitize_sequence(seq))) for seq in seqs
    ]
    encoding = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    )
    return (
        encoding["input_ids"].to(torch_device),
        encoding["attention_mask"].to(torch_device),
    )


def compute_embeddings(backbone, input_ids, attention_mask, max_seq_len: int):
    """Run the backbone and return per-residue embeddings as a ``float32`` array.

    The leading prefix token is stripped and the result is clipped to
    ``max_seq_len`` residues, yielding shape ``(batch, max_seq_len, hidden)``.
    """
    hidden = backbone(input_ids, attention_mask=attention_mask).last_hidden_state
    hidden = hidden * attention_mask.unsqueeze(-1)
    emb = hidden[:, PREFIX_TOKEN_LEN : PREFIX_TOKEN_LEN + max_seq_len, :]
    return emb.float().cpu().numpy()
