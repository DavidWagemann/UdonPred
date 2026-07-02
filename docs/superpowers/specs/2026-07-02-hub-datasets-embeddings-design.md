# Hub datasets & per-pLM embeddings — design

Date: 2026-07-02

## Goal

Distribute UdonPred's training/validation/test data and its per-pLM embeddings
through the Hugging Face Hub, the same way `udonpred/prediction-heads` already
distributes the ONNX heads:

1. Upload the per-target `train/valid/test` FASTA + jsonl files to a single
   public dataset repo and consume them (for training) from there.
2. Upload and consume **precomputed embeddings** produced by different protein
   language models (pLMs), for both training and inference.

`trizod2` is out of scope: its folder layout differs (no `train.fasta`/
`valid.fasta`, plus a pre-built `hf/` arrow `DatasetDict`) and it is handled
separately.

## Decisions (locked in)

- **One dataset repo** `udonpred/datasets` (public), with **per-target configs**.
- **Embeddings live inside the dataset repo**, under a per-pLM subfolder.
- **Training is precomputed-first**: prefer precomputed embeddings from the Hub
  for the configured pLM; fall back to on-the-fly compute (+ optional upload).
- **Inference `--embeddings` accepts a Hub reference** in addition to a local
  file.
- Targets: `trizod, chezod, disprot, pdbflex, plddt, softdis, atlas` (7).

## Key facts discovered

- jsonl schema: `{"id", "y" (labels w/ 999 sentinels), "x_0" (sequence)}`.
- FASTA header is the jsonl `id` with underscores stripped
  (`15036_1_1_1` → `>15036111`). **Embeddings are keyed by the jsonl `id`**, not
  the FASTA header.
- In this dev environment `datasets` + `huggingface_hub` are installed but
  `torch`/`transformers` are not. Dataset upload and all Hub plumbing are
  runnable/testable here; **generating real embeddings needs the `[embedding]`
  stack + GPU** and is a deferred, user-run offline job.

## Repo layout

```
udonpred/datasets/                       (HF dataset repo, public)
├── README.md                            # dataset card w/ `configs:` (7 targets)
├── trizod/
│   ├── train.jsonl  valid.jsonl  test.jsonl
│   ├── train.fasta  valid.fasta  test.fasta
│   └── embeddings/<plm-slug>/{train,valid,test}.h5   # keyed by jsonl id
├── chezod/  disprot/  pdbflex/  plddt/  softdis/  atlas/  …
```

`load_dataset("udonpred/datasets", "trizod")` → `train`/`validation`/`test`
(the `valid.jsonl` file maps to the `validation` split).

## Components

### 1. `udonpred/datasets.py` — lean consume module (hub-optional, no torch, no `datasets` lib)

Mirrors `udonpred/heads.py`: pure path logic + a lazily-imported
`huggingface_hub` download branch, so inference stays lean.

- `DEFAULT_DATASET_REPO = "udonpred/datasets"`, `DEFAULT_DATASET_REVISION = "main"`
  (overridable via arg / `UDONPRED_DATASET_REPO` / `UDONPRED_DATASET_REVISION`).
- `plm_slug(name) -> str`: curated aliases (`Rostlab/ProstT5_fp16` → `prostt5`)
  else sanitize the model id's last path component
  (`facebook/esm2_t33_650M_UR50D` → `esm2-t33-650m-ur50d`).
- `parse_embeddings_ref(source) -> (repo_id, path_in_repo, revision) | None`:
  returns `None` for a local path; otherwise parses `repo_id:path_in_repo`
  with optional `@revision`.
- `resolve_embeddings_ref(source, revision=None) -> Path`: existing local file →
  used as-is; else parsed Hub ref → `hf_hub_download`. Used by inference for
  `--embeddings`.
- `resolve_embeddings(target, split, plm, repo=…, revision=…) -> Path`:
  downloads `<target>/embeddings/<plm>/<split>.h5`.
- `resolve_split_file(target, split, kind="jsonl"|"fasta", repo=…, revision=…) -> Path`.

### 2. `udonpred/utils/publish.py` — upload tooling (needs `[hub]`)

- `publish_dataset(data_dir, repo, targets)`: create the dataset repo, upload
  each target's `{train,valid,test}.{jsonl,fasta}` under `<target>/`, and
  (re)write `README.md` with a `configs:` block (one config per target; splits
  `train`/`validation`/`test` → `train.jsonl`/`valid.jsonl`/`test.jsonl`).
  Exposed as the `udonpred-publish-dataset` console script. **Runnable now.**
- `publish_embeddings(h5_path, repo, target, split, plm)`: upload one `.h5` to
  `<target>/embeddings/<plm>/<split>.h5`. Reused by `udonpred-embed
  --push-to-hub` (adds `--dataset`, `--split`, `--repo`; `plm` derived from the
  backbone). **Generation deferred to a GPU run.**

### 3. Training — precomputed-first (`udonpred/training/model/data.py`)

New `build_datasets` behavior per split:

1. Load the jsonl: from the Hub (`load_dataset(repo, target)`) when a dataset
   Hub source is configured, else the existing local `path` jsonl.
2. Resolve the pLM slug from the backbone name. Try
   `resolve_embeddings(target, split, plm)`; on success, **attach embeddings
   from the `.h5` and skip on-the-fly compute**.
3. Otherwise fall back to the existing SQLite compute path; if
   embedding upload is enabled, save the split's embeddings to `.h5` and
   `publish_embeddings`.

Extract a torch-free helper
`attach_precomputed_embeddings(dataset_split, h5_path, id_column="id") ->
dataset_split` (uses `datasets` + `h5py` + `numpy`) that adds the
`embedding_0` column by looking up `h5[str(id)]`. This is independently
unit-testable without torch.

Config additions (`config/data.yaml` / `config/config.yaml`), all optional and
backwards-compatible:
- global `datasets: {repo: "udonpred/datasets", revision: "main"}`,
- per-dataset `config: <target>` (defaults to the dir basename),
- global `embeddings: {source: "auto"|"compute", upload: false}` (`auto` =
  precomputed-first). pLM comes from `backbone.name`.

### 4. Inference — Hub references for `--embeddings`

`udonpred-caid` and `udonpred-predict` pass `args.embeddings` through
`resolve_embeddings_ref(...)` before loading, so `--embeddings` accepts:
- a local `.h5`/`.npy` path (unchanged), or
- a Hub reference `udonpred/datasets:trizod/embeddings/prostt5/test.h5`
  (optionally `…@revision`).

Only `huggingface_hub` (the `[hub]` extra) is needed; the CAID container stays
otherwise lean. When the local file is passed (as in the baked-in container
flow), no Hub import happens.

## Dependencies

- Consume dataset jsonl for training: `datasets` (already in the `training`
  extra).
- All Hub file downloads (`.h5`, jsonl, fasta) and uploads: `huggingface_hub`
  (the `hub` extra). The lean inference `--embeddings` Hub-ref path needs only
  `hub`.

## Testing

- `test_datasets.py`: `plm_slug` cases; `parse_embeddings_ref` (local / Hub /
  `@rev`); `resolve_embeddings` path building and `resolve_embeddings_ref`
  local short-circuit + Hub branch (monkeypatched `hf_hub_download`);
  `resolve_split_file`.
- `attach_precomputed_embeddings`: tiny in-memory `datasets.Dataset` + tiny
  `.h5` keyed by id → assert `embedding_0` attached and aligned (real
  `datasets`+`h5py`, torch-free).
- `publish.py`: monkeypatched `HfApi` → `publish_dataset` uploads the expected
  files + a card with `configs:`; `publish_embeddings` targets the right path;
  empty/missing inputs raise.
- Boundary: extend the existing guard so the CAID path still imports neither
  `torch` nor `huggingface_hub` at import time (the Hub-ref resolve is lazy).
- Embedding generation and full training runs are **not** exercised (need the
  heavy stack); those tests `importorskip`.

## One-time actions

- **Now**: `publish_dataset` the 7 targets' jsonl+fasta to `udonpred/datasets`
  (small; no torch needed) and write the dataset card.
- **Deferred (user, GPU)**: generate embeddings per pLM and
  `udonpred-embed --push-to-hub` / `udonpred-publish-embeddings` them into the
  repo.

## Out of scope

- `trizod2` (different layout; handled separately).
- Pinning dataset revisions with tags (default `main`; revision override
  exists). Can be added later if reproducibility needs it.
- Any change to how the ONNX prediction heads are distributed.
