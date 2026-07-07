# CAID4-compliant UdonPred predictor.
#
# Inference only: consumes precomputed ProstT5 embeddings (.npy/.h5) and runs
# the bundled ONNX prediction heads on CPU. The heads are pulled from the
# Hugging Face Hub *at build time* and baked into the image, so inference needs
# no network access. No protein language model, no GPU required.

# ---- Stage 1: fetch the pinned prediction heads from the Hub ----------------
FROM python:3.13-slim AS heads

# Overridable at build time, e.g. `--build-arg HEADS_REVISION=main` for latest.
ARG HEADS_REPO=udonpred/prediction-heads
ARG HEADS_REVISION=v0.2.0
ENV HEADS_REPO=${HEADS_REPO} \
    HEADS_REVISION=${HEADS_REVISION}

RUN pip install --no-cache-dir "huggingface_hub>=0.25"
RUN python -c "import os; from huggingface_hub import snapshot_download; \
snapshot_download(os.environ['HEADS_REPO'], revision=os.environ['HEADS_REVISION'], \
local_dir='/weights', allow_patterns=['*.onnx'])"

# ---- Stage 2: lean, offline inference runtime -------------------------------
FROM python:3.13-slim

# No network access at runtime; keep the image self-contained.
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    UDONPRED_HEADS_DIR=/app/weights

WORKDIR /app

# Install the slim inference dependencies only (no huggingface_hub at runtime).
COPY requirements-caid.txt .
RUN pip install --no-cache-dir -r requirements-caid.txt

# Ship the predictor code and the (tiny) ONNX prediction heads.
# The PLM and its embeddings are NOT included — they are provided at runtime.
# .dockerignore keeps the heavy udonpred/{embedding,training,utils} subpackages
# out of the build context, so only the lean torch-free core is copied. The
# package lives under src/ but is flattened to /app/udonpred in the image.
COPY src/udonpred/ ./udonpred/
COPY --from=heads /weights ./weights

# Heads are baked in at /app/weights (UDONPRED_HEADS_DIR), consumed locally with
# no network. Override the model source via the positional model_dir argument.
ENTRYPOINT ["python", "-m", "udonpred.caid.predict"]
CMD ["--help"]
