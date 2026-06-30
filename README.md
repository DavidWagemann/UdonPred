# UdonPred: Untangling Protein Intrinsic Disorder Prediction
<img align="right" src="images/udonpred_logo_small.png" alt="image" height="20%" width="20%" />

**UdonPred** is a set of models for predicting disorder based on different definitions of disorder, reaching state-of-the-art performance using simple, [ProstT5](https://github.com/mheinzinger/ProstT5)-based embeddings. 
The ability of models trained on one definition of disorder to generalise to others was accessed in the corresponding [publication](https://doi.org/10.64898/2026.01.26.701679).

The training data can be found on [here](https://doi.org/10.6084/m9.figshare.31444642).

For the old version submitted to CAID3 go to [https://github.com/jschlensok/udonpred](https://github.com/jschlensok/udonpred).

## Usage
### Docker
The quickest way to run UdonPred is via Docker. The image is inference-only: it
bundles the predictor code and the small ONNX prediction heads, and consumes
**precomputed ProstT5 embeddings** (`.npy`/`.h5`) on CPU.

Build the image:
```
docker build -t udonpred .
```

Run a prediction (mount your data into `/data`):
```
docker run --rm -v "$PWD":/data udonpred \
    /data/input.fasta /app/weights --embeddings /data/embeddings.h5 \
    --target all --threads 24 --output /data/out
```

Generate the embeddings beforehand (outside the container) with
ProstT5 model `Rostlab/ProstT5_fp16`; see [EMBEDDINGS.md](EMBEDDINGS.md) for the
full specification:
```
python embed.py input.fasta --output embeddings.h5
```

Options:
- `--target` — one or more of trizod/chezod/softdis/pdbflex/atlas/plddt/disprot,
  or `all` to run every head in `weights/` (default: trizod).
- `--embeddings` (required), `--output` (dir; stdout if unset),
  `--device` (cpu/cuda, default cpu), `--threads` (CPU thread cap),
  `--smooth` (Gaussian sigma, 0 to disable).

**Output:** one CAID file **per prediction head**, named `udonpred_<target>.caid`, each
holding the predictions for **all** input proteins concatenated, written flat into the
output directory (e.g. `out/udonpred_trizod.caid`, `out/udonpred_disprot.caid`).

#### How to Generate Embeddings
**Important note:** The CAID4 predictor (`caid/predict.py`) does **not** run the protein language
model. It consumes per-residue **ProstT5** embeddings that are precomputed and
passed in via `--embeddings`. This document is the exact specification for
generating them.

##### pLM Specifications

| Property | Value |
| --- | --- |
| Model | `Rostlab/ProstT5_fp16` (T5 encoder) |
| Direction | AA → 3Di, i.e. the `<AA2fold>` prefix token |
| Representation | `last_hidden_state` (encoder output) |
| Embedding dimension | **1024** |
| dtype | `float32` (fp16 is accepted; the predictor up-casts) |
| Ambiguous residues | `B, Z, J, U, O, *` → `X` before tokenisation |

##### Input Format
The predictor needs **one row per residue**: shape `(L, 1024)` for a sequence
of length `L`. The predictor is tolerant of the ProstT5 special tokens and will
trim them automatically, accepting any of:

- `(L, 1024)`   — already trimmed (preferred)
- `(L+1, 1024)` — leading `<AA2fold>` prefix included
- `(L+2, 1024)` — leading prefix **and** trailing `</s>` (EOS) included

Any other length is treated as a mismatch and raises an error.

Expected format: **`.h5`** — one dataset per sequence, keyed by the **FASTA header** (the text
  after `>`, whitespace-trimmed). Use this for multi-sequence FASTA files.

##### Usage
The repository ships `embed.py`, which produces embeddings in exactly the
expected layout (trimmed to `(L, 1024)`, float32, ambiguous residues mapped to
`X`):

```bash
# Multi-sequence -> HDF5 keyed by FASTA header
python embed.py input.fasta --output embeddings.h5
```

`embed.py` requires `torch` and `transformers` (see `pyproject.toml`); it is the
only component that downloads/loads ProstT5 and is intentionally **outside** the
CAID inference container.

### Manually
1. `git clone https://github.com/DavidWagemann/UdonPred.git`
2. `cd UdonPred`
3. `uv sync`
4. `uv run predict.py {path to fasta} {path to weights}`

You can use the following options:
- `--target`: chooses the model trained on the specified dataset (trizod, chezod, softdis, pdbflex, atlas, plddt, disprot). The default is trizod.
- `--output`: sets the output directory path. Each sequence will be saved as a .caid file. The output will be written to the terminal if this is not set.
- `--batch-size`: sets the total sequence length per batch. Try reducing this if you get an out of memory error.
- `--device`: sets the device used for inference (cpu or cuda). Uses cuda by default if available.
- `--smooth`: Applies gaussian smoothing with the give sigma to the results in order to remove prediction noise. 

## Retraining
UdonPred can be retrained by placing the required data as jsonl files in a data/ subfolder and pointing to it in `config/data.yaml`. The training configuration and architecture can be changed in `config/config.yaml` and `config/architecture.yaml` respectively. To start the training process, run `uv run run.py train`. 

After training is complete, a checkpoint can be exported for use with the prediction script using `uv run export.py {path to checkpoint}`. For export options, see `uv run export.py --help`.

## How to Cite
```
@article {UdonPred,
	author = {Schlensok, Julius and Wagemann, David and Senoner, Tobias and Haak, Markus and Rost, Burkhard},
	title = {UdonPred: Untangling Protein Intrinsic Disorder Prediction},
	elocation-id = {2026.01.26.701679},
	year = {2026},
	doi = {10.64898/2026.01.26.701679},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/01/28/2026.01.26.701679},
	eprint = {https://www.biorxiv.org/content/early/2026/01/28/2026.01.26.701679.full.pdf},
	journal = {bioRxiv}
}
```
