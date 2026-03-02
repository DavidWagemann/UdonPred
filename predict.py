import argparse
import re
import sys
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from tqdm import tqdm


def read_fasta(path: str) -> List[Tuple[str, str]]:
    """Read FASTA file and return list of (header, sequence) tuples."""
    entries: List[Tuple[str, str]] = []
    header = None
    seq_chunks: List[str] = []

    with open(path, "r") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(re.sub(r"\s+", "", line))

    if header is not None:
        entries.append((header, "".join(seq_chunks)))

    return entries


def load_config(model_dir: str | Path, config_name: str = "config") -> Dict:
    """Load configuration from export directory."""
    export_path = Path(model_dir)
    config_path = export_path / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_name}.yaml not found at: {config_path}")
    with config_path.open("r") as f:
        return yaml.safe_load(f)

def format_predictions(
    header: str,
    sequence: str,
    scores,
) -> List[str]:
    lines: List[str] = [f">{header}\n"]

    for idx, (aa, row) in enumerate(zip(sequence, scores), start=1):
        if hasattr(row, "__len__") and not isinstance(row, str):
            val = row[0] if len(row) > 0 else 0.0
        else:
            val = row
        lines.append(f"{idx}\t{aa}\t{float(val):.3f}\t\n")
    return lines


def resolve_device(torch, device: str) -> str:
    """Resolve device string to actual device."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "cuda"
    if device == "cpu":
        return "cpu"
    raise ValueError("Device must be one of: auto, cpu, cuda")


def resolve_device_dtype(torch, device: str) -> Tuple[str, str]:
    """Resolve device and dtype for inference."""
    torch_device = resolve_device(torch, device)
    dtype = "float16" if torch_device == "cuda" else "float32"
    return torch_device, dtype


def tokenize_batch(embedder, seqs: List[str], torch_device: str):
    input_ids, attention_mask = embedder.tokenise(seqs)
    return input_ids.to(torch_device), attention_mask.to(torch_device)


def run_backbone(
    backbone, input_ids, attention_mask, prefix_len: int, max_seq_len: int
):
    emb = backbone(input_ids, attention_mask)
    emb = emb * attention_mask.unsqueeze(-1)
    return emb[:, prefix_len : prefix_len + max_seq_len, :]

def iter_batches(items: List[Tuple[str, str]], max_total_len: int):
    batch: List[Tuple[str, str]] = []
    total_len = 0
    for header, seq in items:
        seq_len = len(seq)
        if batch and total_len + seq_len > max_total_len:
            yield batch
            batch = []
            total_len = 0
        batch.append((header, seq))
        total_len += seq_len
        if total_len >= max_total_len:
            yield batch
            batch = []
            total_len = 0
    if batch:
        yield batch

def count_batches(items: List[Tuple[str, str]], max_total_len: int) -> int:
    count = 0
    total_len = 0
    for _, seq in items:
        seq_len = len(seq)
        if count == 0 and total_len == 0 and seq_len == 0:
            continue
        if total_len and total_len + seq_len > max_total_len:
            count += 1
            total_len = 0
        total_len += seq_len
        if total_len >= max_total_len:
            count += 1
            total_len = 0
    if total_len:
        count += 1
    return count


def run_exported(
    entries: List[Tuple[str, str]],
    model_dir: str,
    target: str,
    config: Dict,
    max_total_seq_len: int,
    output_path: str | None,
    device: str,
) -> None:
    """Run prediction using exported models."""
    torch = import_module("torch")
    Embedder = getattr(import_module("model.embedder"), "Embedder")

    torch_device, dtype = resolve_device_dtype(torch, device)

    bb_params = config["config"]["backbone"]

    export_path = Path(model_dir)
    backbone_path = export_path / f"backbone_{dtype}.pt2"
    if not backbone_path.exists():
        raise FileNotFoundError(f"Exported backbone not found: {backbone_path}")

    backbone = torch.export.load(backbone_path).module()

    head_path = export_path / f"{target}_{dtype}.pt2"
    if not head_path.exists():
        raise FileNotFoundError(f"Exported head not found: {head_path}")
    head = torch.export.load(head_path).module()

    embedder = Embedder(
        backbone_name=bb_params["name"],
        prefix_token=bb_params["prefix_token"],
        tokeniser_type=bb_params["tokeniser_type"],
        model_type=bb_params["model_type"],
    )
    embedder.load_tokeniser()
    prefix_token_len = embedder.prefix_token_len

    if output_path:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    with torch.inference_mode():
        total_batches = count_batches(entries, max_total_seq_len)
        for batch in tqdm(
            iter_batches(entries, max_total_seq_len), total=total_batches
        ):
            headers, seqs = zip(*batch)
            seqs = list(seqs)
            max_seq_len = max(len(s) for s in seqs)
            input_ids, attention_mask = tokenize_batch(embedder, seqs, torch_device)

            emb = run_backbone(
                backbone, input_ids, attention_mask, prefix_token_len, max_seq_len
            )

            predictions = head(emb)
            batch_scores = predictions.cpu().numpy()

            for header, seq, scores in zip(headers, seqs, batch_scores):
                scores_seq = scores[: len(seq)]
                formatted_lines = format_predictions(
                    header, seq, scores_seq
                )
                if out_dir:
                    safe_header = header.replace("/", "_").replace("|", "_")
                    file_path = out_dir / f"{safe_header}.caid"
                    with file_path.open("w") as f:
                        f.writelines(formatted_lines)
                else:
                    sys.stdout.writelines(formatted_lines)


def collect_output_keys(config: Dict) -> List[str]:
    """Collect output keys from configuration."""
    output_keys = set()
    for ds in config.get("data", {}):
        if config["data"][ds].get("fraction", 0) > 0:
            keys = set()
            if "losses" in config["data"][ds]:
                for key_losses in config["data"][ds]["losses"].values():
                    for loss in key_losses:
                        keys.add(loss["output"])
            if "metrics" in config["data"][ds]:
                for key_metrics in config["data"][ds]["metrics"].values():
                    for metric in key_metrics:
                        keys.add(metric["output"])
            output_keys |= keys
    outputs = config.get("config", {}).get("outputs")
    if outputs:
        return list(outputs)
    return sorted(output_keys)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict using exported optimized models for faster inference."
    )
    parser.add_argument("fasta", type=str, help="Path to input FASTA file")
    parser.add_argument("model_dir", type=str, help="Directory with exported models")
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default="trizod",
        choices=["trizod", "chezod", "softdis", "pdbflex", "atlas", "plddt", "disprot"],
        help="Prediction Type (selects model trained on the specified dataset)"

    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory path (will save each sequence as a .caid file)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=2000,
        help="Max total sequence length per batch (dynamic batching)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference (auto, cpu, cuda). Dtype will be selected automatically.",
    )


    args = parser.parse_args()

    entries = read_fasta(args.fasta)
    if not entries:
        raise ValueError("No FASTA entries found.")

    entries = sorted(entries, key=lambda item: len(item[1]), reverse=True)

    if not Path(args.model_dir).is_dir():
        raise ValueError("Export directory not found.")

    config = load_config(args.model_dir, args.target)

    run_exported(
        entries,
        args.model_dir,
        args.target,
        config,
        args.batch_size,
        args.output,
        args.device,
    )


if __name__ == "__main__":
    main()
