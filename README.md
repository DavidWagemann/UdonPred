# UdonPred: Untangling Protein Intrinsic Disorder Prediction
<img align="right" src="images/udonpred_logo_small.png" alt="image" height="20%" width="20%" />

**UdonPred** is a set of models for predicting disorder based on different definitions of disorder, reaching state-of-the-art performance using simple, [ProstT5](https://github.com/mheinzinger/ProstT5)-based embeddings. 
The ability of models trained on one definition of disorder to generalize to others was accessed in the corresponding [publication](https://doi.org/10.64898/2026.01.26.701679).

The training data can be found on [here](https://doi.org/10.6084/m9.figshare.31444642).

For the old version submitted to CAID3 go to [https://github.com/jschlensok/udonpred](https://github.com/jschlensok/udonpred).

## Usage
### Docker
The quickest way to run UdonPred is via Docker. The image is inference-only: it
consumes **precomputed ProstT5 embeddings** (`.h5`) on CPU. The ONNX prediction
heads are pulled from the Hugging Face Hub
([`udonpred/prediction-heads`](https://huggingface.co/udonpred/prediction-heads))
**at build time** and baked into the image, so inference needs no network access.

Build the image (pins the released heads; pass `--build-arg HEADS_REVISION=main`
for the latest, or another tag to pin a different release):
```
docker build -t udonpred .
```

Run a prediction (mount your data into `/data`). The heads live at `/app/weights`
inside the image and are used automatically, so `model_dir` can be omitted:
```
docker run --rm -v "$PWD":/data udonpred \
    /data/input.fasta --embeddings /data/embeddings.h5 \
    --target all --threads 24 --output /data/out
```

Generate the embeddings beforehand (outside the container) with
ProstT5 model `Rostlab/ProstT5_fp16`; see [EMBEDDINGS.md](EMBEDDINGS.md) for the
full specification:
```
uv run --extra embedding udonpred-embed input.fasta --output embeddings.h5
```

Options:
- `--target` — one or more of trizod/chezod/softdis/pdbflex/atlas/plddt/disprot,
  or `all` to run every head in `weights/` (default: trizod).
- `--embeddings` (required), `--output` (dir; stdout if unset),
  `--device` (cpu/cuda, default cpu), `--threads` (CPU thread cap),
  `--smooth` (Gaussian sigma, 0 to disable).
- `--normalize` / `--no-normalize` — map scores onto the CAID convention
  (default: enabled). See below.

**Output:** one directory **per prediction head** (the CAID "flavor"), holding one file
**per protein** — `<output>/<target>/<protein>.caid`. Each row is
`<index>\t<residue>\t<score>\t<binary>`:

```
out/chezod/P04637.caid
    >P04637
    1	M	0.892	1
    2	E	0.813	1
    ...
```

Filenames come from the FASTA header with path separators, pipes, and whitespace replaced by
`_`, so distinct headers never collide. Without `--output`, all predictions go to stdout
instead, prefixed by a `# target: <name>` line when more than one head is running.

#### Timings

Every flavor directory also gets a `timings.csv` recording how long that head took per
protein:

```
out/chezod/timings.csv
    # Running UdonPred
    sequence,milliseconds
    P04637,1827
```

The measured interval covers the per-protein work only — embedding alignment (or, for
`udonpred-predict`, tokenization and the ProstT5 forward pass), ONNX scoring, smoothing,
binarization, and writing the `.caid` file. One-time setup (reading the FASTA, loading the
heads and backbone, opening the embedding file) is excluded. Embedding preparation is done
once per protein and shared across heads, so each head is charged for it — running one flavor
alone would still pay that cost. Protein names come from the FASTA header, quoted if they
contain a comma. The banner deliberately carries no timestamp, so re-running over the same
input produces byte-identical files. Nothing is written when predictions go to stdout, since
there is no directory to put the file in.

#### Score Normalization and Binarization

CAID expects a score in `[0, 1]` where **higher means more disordered**, plus a binary
disorder call. The heads do not natively produce either: four end in a sigmoid, while the rest
are unbounded regressions on their target's own scale — and `chezod`/`plddt` run the *other*
way (higher = more ordered). Both columns are derived from one per-head policy table,
`TARGET_POLICIES` in [`src/udonpred/inference.py`](src/udonpred/inference.py):

| head | raw output | higher means | score transform | disordered when |
| --- | --- | --- | --- | --- |
| `trizod`, `trizod2` | `[0, 1]` (sigmoid) | disorder | none | `x ≥ 0.4` |
| `disprot` | `[0, 1]` (sigmoid) | disorder | none | `x ≥ 0.5` |
| `softdis` | `[0, 1]` (sigmoid) | disorder | none | `x ≥ 0.025` |
| `chezod` | Z-score, unbounded | order | `1 - (clamp(x, -5, 16.15) + 5) / 21.15` | `x < 3` |
| `plddt` | pLDDT, unbounded | order | `1 - clamp(x, 0, 100) / 100` | `x < 68.8` |
| `pdbflex` | Å RMSD, unbounded | disorder | `clamp(x, 0, 10) / 10` | `x ≥ 2` |
| `atlas` | Å RMSF, unbounded | disorder | `clamp(x, 0, 10) / 10` | `x ≥ 2` |

Thresholds are stated in each head's **raw** units, so they stay readable against the
literature cutoffs and the binary column is identical with or without `--normalize`. Order is:
smooth on the raw scale → read off the binary calls → rescale the score column.

Score bounds are fixed rather than derived from the input's observed min/max, so a residue's
score does not depend on which other proteins were in the same run. Pass `--no-normalize` to
keep raw scores in the third column — useful for regression analysis, but not CAID-compliant.
A head with no entry in the table is rejected up front, since its threshold is needed either
way.

#### How to Generate Embeddings
**Important note:** The CAID4 predictor (`udonpred-caid`, i.e. `udonpred.caid.predict`) does **not** run the protein language
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
| Ambiguous residues | `B, Z, J, U, O, *` → `X` before tokenization |

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
The repository ships the `udonpred-embed` command
(`udonpred.utils.embed`), which produces embeddings in exactly the
expected layout (trimmed to `(L, 1024)`, float32, ambiguous residues mapped to
`X`):

```bash
# Multi-sequence -> HDF5 keyed by FASTA header
uv run --extra embedding udonpred-embed input.fasta --output embeddings.h5
```

`udonpred-embed` requires the `embedding` extra (`torch` + `transformers`;
`uv sync --extra embedding`); it is the only component that downloads/loads
ProstT5 and is intentionally **outside** the CAID inference container.

### Manually
1. `git clone https://github.com/davidwagemann/udonpred.git .`
2. `uv sync --extra embedding` (the on-the-fly predictor needs `torch` + `transformers`; plain `uv sync` installs only the lean inference core)
3. `uv run udonpred-predict {path to fasta}`

The `model_dir` argument is optional: omit it to pull the released ONNX heads
from the Hub ([`udonpred/prediction-heads`](https://huggingface.co/udonpred/prediction-heads),
cached locally), or pass a **local directory** to use your own heads offline, or
a **Hub repo id** to pull a different set. Fetching from the Hub needs the `hub`
extra (`uv sync --extra hub`; bundled with `embedding`/`training`/dev).

This runner shares its flags and its output format with `udonpred-caid` — see
[Score Normalization and Binarization](#score-normalization-and-binarization) above,
which applies identically here. Options:
- `--target`: one or more models trained on the given datasets (trizod, chezod, softdis, pdbflex, atlas, plddt, disprot), or `all` to run every head. The default is trizod.
- `--revision`: Hub revision (tag/branch/commit) to pull the heads from when `model_dir` is a repo id or omitted (default: the pinned release).
- `--output`: output directory. Each head gets a `<target>/` subdirectory holding one `<protein>.caid` file per input protein. The output goes to the terminal if this is not set.
- `--normalize` / `--no-normalize`: map the score column onto the CAID `[0, 1]` disorder convention (default: enabled).
- `--batch-size`: sets the total sequence length per batch. Try reducing this if you get an out of memory error.
- `--device`: sets the device used for inference (auto, cpu, or cuda). Uses cuda by default if available.
- `--smooth`: Applies gaussian smoothing with the given sigma to the results in order to remove prediction noise.

## Retraining
Install the training stack with `uv sync --extra training`. UdonPred can be retrained by placing the required data as jsonl files in a data/ subfolder and pointing to it in `config/data.yaml`. The training configuration and architecture can be changed in `config/config.yaml` and `config/architecture.yaml` respectively. To start the training process, run `uv run udonpred-train train`.

To restrict training (and its validation/test metrics) to longer proteins, set `min_length` in `config/config.yaml` (residues; `0` keeps all, e.g. `50` or `100` to drop short peptides). The same threshold is available when evaluating existing heads: `uv run --extra hub python eval_trizod_heads.py --min-length 50`.

### Multi-target orchestration
`scripts/` bundles the launchers that train each `(target × pLM)` combination
separately in a GPU container, sharing `scripts/run_training.sh` (it runs
`uv sync` + `udonpred-train train` inside whatever container invokes it):

- **Slurm** (`scripts/slurm/train.sbatch`): a job array on the cluster via
  enroot/pyxis. Submit with `sbatch scripts/slurm/train.sbatch`. Export
  `WANDB_API_KEY` in your login shell first (the container runs
  non-interactively, so `~/.netrc` is never read).
- **Docker** (`scripts/docker/train.sh`): the same matrix, sequentially, via
  `docker run --gpus`. Overridable env: `IMAGE`, `MOUNT`, `GPUS`, `SCRATCH`,
  `WANDB_MODE`.

The base jsonl datasets are gitignored, so a fresh checkout has an empty
`data/`. Fetch them from the [`udonpred/datasets`](https://huggingface.co/datasets/udonpred/datasets)
Hub repo with `scripts/fetch_data.sh` (downloads every target's jsonl+fasta into
`data/split/`; overridable `REPO`/`DEST`). Run it once before training.

With a **root Docker daemon**, run the project from a **local disk**, not an
NFS home — a root-squashed NFS mount can't be bind-mounted by the daemon (and
outputs can't be written back to it). The script fails fast if it detects an
NFS project. `scripts/docker/stage.sh` copies a working tree onto local scratch
for you and run from there:

```bash
scripts/fetch_data.sh            # download the (gitignored) jsonl datasets into data/split/
scripts/docker/stage.sh          # rsync repo → /mnt/space/local/UdonPred (DEST= to override)
cd /mnt/space/local/UdonPred && scripts/docker/train.sh
# checkpoints land on local disk; copy them back to NFS from your shell afterwards
```

`stage.sh` skips `.git`/`.venv`, the stray root `frustraiseq_embeddings.h5`, and
the unused `trizod2` embeddings, so only the ~1.3 GB of frustraiseq embeddings
the matrix needs are copied. The prostt5 embeddings aren't local — they're
pulled from the Hub into the persistent HF cache under `SCRATCH` on first use.

(A rootless Docker/enroot runtime, which keeps your UID, avoids the NFS issue
entirely.)

After training is complete, export the checkpoint to ONNX and publish the heads
to the Hub in one step:
```
uv run udonpred-export {checkpoint-root} -o exported/ --push-to-hub --tag v0.2.0
```
This uploads every exported `*.onnx` head to `udonpred/prediction-heads` (override
with `--hf-repo`) using your ambient `huggingface-cli login`, and tags the release.
Remember to bump `DEFAULT_HEADS_REVISION` in `src/udonpred/heads.py` (and the
`HEADS_REVISION` build arg in the `Dockerfile`) to the new tag so inference and
the container pull the new heads by default. Omit `--push-to-hub` to export
locally only. For all options, see `uv run udonpred-export --help`.

### Datasets & embeddings on the Hub
The per-target datasets live in the public dataset repo
[`udonpred/datasets`](https://huggingface.co/datasets/udonpred/datasets), one
config per target (`trizod2` excluded):

```python
from datasets import load_dataset
ds = load_dataset("udonpred/datasets", "trizod")   # train / validation / test
```

Each target holds `train/valid/test` as jsonl (`{id, y, x_0}`) + FASTA, and
precomputed per-pLM embeddings at `<target>/embeddings/<plm>/<split>.h5` (keyed
by the jsonl `id`).

- **Publish datasets:** `uv run udonpred-publish-dataset data/split` uploads
  every target's jsonl+fasta and (re)writes the dataset card.
- **Precomputed-first training:** with `config.embeddings.source: auto`,
  `build_datasets` pulls embeddings for the backbone pLM from the Hub when
  available and only computes on the fly as a fallback.
- **Publish embeddings** (needs the `embedding` extra + a GPU):
  `uv run --extra embedding udonpred-embed <target>/test.fasta -o test.h5
  --push-to-hub --dataset <target> --split test`.
- **Consume embeddings at inference:** `--embeddings` accepts a Hub reference,
  e.g. `--embeddings udonpred/datasets:trizod/embeddings/prostt5/test.h5`
  (optionally `…@revision`), in addition to a local `.h5`/`.npy`.

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
