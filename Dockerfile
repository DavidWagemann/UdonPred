# CAID4-compliant UdonPred predictor.
#
# Inference only: consumes precomputed ProstT5 embeddings (.npy/.h5) and runs
# the bundled ONNX prediction heads on CPU. No protein language model, no
# network access, no GPU required.
FROM python:3.13-slim

# No network access at runtime; keep the image self-contained.
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# Install the slim inference dependencies only.
COPY requirements-caid.txt .
RUN pip install --no-cache-dir -r requirements-caid.txt

# Ship the predictor code and the (tiny) ONNX prediction heads.
# The PLM and its embeddings are NOT included — they are provided at runtime.
# .dockerignore keeps the heavy udonpred/{embedding,training,utils} subpackages
# out of the build context, so only the lean torch-free core is copied. The
# package lives under src/ but is flattened to /app/udonpred in the image.
COPY src/udonpred/ ./udonpred/
COPY weights/ ./weights/

# Default model directory; override paths via CLI arguments.
ENTRYPOINT ["python", "-m", "udonpred.caid.predict"]
CMD ["--help"]
