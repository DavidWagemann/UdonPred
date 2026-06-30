# UdonPred: Untangling Protein Intrinsic Disorder Prediction
<img align="right" src="images/udonpred_logo_small.png" alt="image" height="20%" width="20%" />

**UdonPred** is a set of models for predicting disorder based on different definitions of disorder, reaching state-of-the-art performance using simple, [ProstT5](https://github.com/mheinzinger/ProstT5)-based embeddings. 
The ability of models trained on one definition of disorder to generalise to others was accessed in the corresponding [publication](https://doi.org/10.64898/2026.01.26.701679).

The training data can be found on [here](https://doi.org/10.6084/m9.figshare.31444642).

For the old version submitted to CAID3 go to [https://github.com/jschlensok/udonpred](https://github.com/jschlensok/udonpred).

## Installation
1. `git clone https://github.com/DavidWagemann/UdonPred.git`
2. `cd UdonPred`
3. `uv sync`

## Usage
`uv run predict.py {path to fasta} {path to weights}`

You can use the following options:
- `--target`: chooses the model trained on the specified dataset (trizod, chezod, softdis, pdbflex, atlas, plddt, disprot). The default is trizod.
- `--output`: sets the output directory path. Each sequence will be saved as a .caid file. The output will be written to the terminal if this is not set.
- `--batch-size`: sets the total sequence length per batch. Try reducing this if you get an out of memory error.
- `--device`: sets the device used for inference (cpu or cuda). Uses cuda by default if available.
- `--smooth`: Applies gaussian smoothing with the give sigma to the results in order to remove prediction noise. 

## CAID4 / Precomputed-embedding Predictor
For CAID4 (CPU-only, offline, no protein language model in the container), use
the dedicated runner `caid/predict.py`. It consumes **precomputed ProstT5
embeddings** instead of running the PLM itself.

### Install (inference only)
```
pip install -r requirements-caid.txt
```
This slim dependency set excludes `torch`/`transformers`.

### 1. Generate embeddings (outside the container)
See [EMBEDDINGS.md](EMBEDDINGS.md) for the full specification. The exact command:
```
python embed.py input.fasta --output embeddings.h5 --device cpu
```
ProstT5 model: `Rostlab/ProstT5_fp16`. Embeddings are provided as `.npy`
(single sequence) or `.h5` (keyed by FASTA header), dimension 1024.

### 2. Predict
```
python -m caid.predict input.fasta weights/ --embeddings embeddings.h5 \
    --target all --threads 24 --output out/
```
Options:
- `--target` — one or more of trizod/chezod/softdis/pdbflex/atlas/plddt/disprot,
  or `all` to run every head in `weights/` (default: trizod).
- `--embeddings` (required), `--output` (dir; stdout if unset),
  `--device` (cpu/cuda, default cpu), `--threads` (CPU thread cap),
  `--smooth` (Gaussian sigma, 0 to disable).

**Output:** one CAID file **per prediction head**, named `<target>.caid`, each
holding the predictions for **all** input proteins concatenated. All files are
written flat into the output directory, e.g. `out/trizod.caid`, `out/disprot.caid`.
The embedding for each protein is aligned once and reused across every head.

### Docker
```
docker build -t udonpred-caid .
docker run --rm -v "$PWD":/data udonpred-caid \
    /data/input.fasta /app/weights --embeddings /data/embeddings.h5 \
    --target trizod --threads 24 --output /data/out
```
The image bundles only the code and the small ONNX heads — never the PLM or its
weights.

## Training a Model
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
