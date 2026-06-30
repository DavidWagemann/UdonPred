# Precomputed embeddings for the CAID4 UdonPred predictor

The CAID4 predictor (`caid/predict.py`) does **not** run the protein language
model. It consumes per-residue **ProstT5** embeddings that are precomputed and
passed in via `--embeddings`. This document is the exact specification for
generating them.

## Model and representation

| Property | Value |
| --- | --- |
| Model | `Rostlab/ProstT5_fp16` (T5 encoder) |
| Direction | AA → 3Di, i.e. the `<AA2fold>` prefix token |
| Representation | `last_hidden_state` (encoder output) |
| Embedding dimension | **1024** |
| dtype | `float32` (fp16 is accepted; the predictor up-casts) |
| Ambiguous residues | `B, Z, J, U, O, *` → `X` before tokenisation |

## Per-sequence array shape

The predictor needs **one row per residue**: shape `(L, 1024)` for a sequence
of length `L`. The predictor is tolerant of the ProstT5 special tokens and will
trim them automatically, accepting any of:

- `(L, 1024)`   — already trimmed (preferred)
- `(L+1, 1024)` — leading `<AA2fold>` prefix included
- `(L+2, 1024)` — leading prefix **and** trailing `</s>` (EOS) included

Any other length is treated as a mismatch and raises an error.

## File formats

- **`.h5`** — one dataset per sequence, keyed by the **FASTA header** (the text
  after `>`, whitespace-trimmed). Use this for multi-sequence FASTA files.
- **`.npy`** — a single `(L, 1024)` array for a single-sequence FASTA.

## Exact command

The repository ships `embed.py`, which produces embeddings in exactly the
expected layout (trimmed to `(L, 1024)`, float32, ambiguous residues mapped to
`X`):

```bash
# Multi-sequence -> HDF5 keyed by FASTA header
python embed.py input.fasta --output embeddings.h5

# Single sequence -> .npy
python embed.py single.fasta --output embeddings.npy
```

`embed.py` requires `torch` and `transformers` (see `pyproject.toml`); it is the
only component that downloads/loads ProstT5 and is intentionally **outside** the
CAID inference container.
