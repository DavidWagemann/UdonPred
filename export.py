import argparse
import os
import shutil
from importlib import import_module
from typing import Dict, List, Tuple

import torch
import yaml


def load_config(checkpoint_dir: str) -> Dict:
    """Load configuration from checkpoint directory."""
    checkpoint_config = os.path.join(checkpoint_dir, "config.yaml")
    if not os.path.exists(checkpoint_config):
        raise FileNotFoundError(
            f"Checkpoint config.yaml not found at: {checkpoint_config}"
        )
    with open(checkpoint_config, "r") as f:
        return yaml.safe_load(f)


def load_hyperparameters(config: Dict) -> Dict:
    """Load hyperparameters if specified in config."""
    hyperparam_path = config.get("config", {}).get("hyperparameter_path")
    if hyperparam_path and os.path.exists(hyperparam_path):
        with open(hyperparam_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def collect_output_keys(config: Dict) -> List[str]:
    """Collect output keys from configuration."""
    output_keys = set()
    for _, ds_config in config.get("data", {}).items():
        if ds_config.get("fraction", 0) > 0:
            keys = set()
            if "losses" in ds_config:
                for key_losses in ds_config["losses"].values():
                    for loss in key_losses:
                        keys.add(loss["output"])
            if "metrics" in ds_config:
                for key_metrics in ds_config["metrics"].values():
                    for metric in key_metrics:
                        keys.add(metric["output"])
            output_keys |= keys
    outputs = config.get("config", {}).get("outputs")
    if outputs:
        return list(outputs)
    return sorted(output_keys)


def get_device_and_dtype(dtype: str) -> Tuple[str, torch.dtype]:
    """Return device string and torch dtype for export."""
    if dtype == "float16":
        return "cuda", torch.float16
    return "cpu", torch.float32


def export_backbone(
    checkpoint_dir: str,
    config: Dict,
    output_dir: str,
    dtype: str = "float32",
) -> None:
    """Export backbone embedder using torch.export.

    Args:
        checkpoint_dir: Path to checkpoint directory
        config: Model configuration
        output_dir: Directory to save exported model
        dtype: Data type for export (float32 or float16)
    """
    device, torch_dtype = get_device_and_dtype(dtype)
    print(f"Exporting backbone to {output_dir} (device: {device}, dtype: {dtype})...")

    Embedder = getattr(import_module("model.embedder"), "Embedder")

    bb_params = config["config"]["backbone"]
    embedder = Embedder(
        backbone_name=bb_params["name"],
        prefix_token=bb_params["prefix_token"],
        tokeniser_type=bb_params["tokeniser_type"],
        model_type=bb_params["model_type"],
    )
    embedder.load_embedder()

    adapter_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if os.path.exists(adapter_path):
        embedder.model.load_adapter(checkpoint_dir, "default")

    embedder.eval()
    embedder = embedder.to(device).to(torch_dtype)

    batch_size = 2
    example_seq_len = 128
    example_input_ids = torch.randint(
        0, 1000, (batch_size, example_seq_len), dtype=torch.long, device=device
    )
    example_attention_mask = torch.ones(
        (batch_size, example_seq_len), dtype=torch.long, device=device
    )

    with torch.inference_mode():
        dynamic_shapes = (
            {0: torch.export.Dim("batch"), 1: torch.export.Dim("seq_len")},  # input_ids
            {
                0: torch.export.Dim("batch"),
                1: torch.export.Dim("seq_len"),
            },
        )
        exported_program = torch.export.export(
            embedder,
            (example_input_ids, example_attention_mask),
            dynamic_shapes=dynamic_shapes,
        )

        backbone_path = os.path.join(output_dir, f"backbone_{dtype}.pt2")
        torch.export.save(exported_program, backbone_path)
        print(f"Backbone exported to {backbone_path}")


def export_heads(
    checkpoint_dir: str,
    config: Dict,
    output_dir: str,
    dtype: str = "float32",
) -> None:
    """Export prediction heads using torch.export.

    Args:
        checkpoint_dir: Path to checkpoint directory
        config: Model configuration
        output_dir: Directory to save exported model
        dtype: Data type for export (float32 or float16)
    """
    device, torch_dtype = get_device_and_dtype(dtype)
    print(
        f"Exporting prediction heads to {output_dir} (device: {device}, dtype: {dtype})..."
    )

    build_prediction_heads = getattr(
        import_module("model.build_model"), "build_prediction_heads"
    )
    UdonPred = getattr(import_module("model.model"), "UdonPred")

    hyperparameters = load_hyperparameters(config)
    prediction_heads = build_prediction_heads(hyperparameters, config)
    output_keys = set(collect_output_keys(config))

    model = UdonPred(None, prediction_heads, output_keys)

    state_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    model.load_state_dict(
        torch.load(state_path, weights_only=True, map_location="cpu"),
        strict=False,
    )

    model.eval()

    model = model.to(device).to(torch_dtype)
    hidden_dim = config["config"]["input_dim"]

    batch_size = 2
    example_seq_len = 128
    example_embeddings = torch.randn(
        batch_size, example_seq_len, hidden_dim, dtype=torch_dtype, device=device
    )

    exported_heads = []

    for head_name, prediction_head in model.prediction_heads.items():
        print(f"  Exporting head: {head_name}")
        prediction_head.eval()

        with torch.no_grad():
            dynamic_shapes = (
                {0: torch.export.Dim("batch"), 1: torch.export.Dim("seq_len")},
            )
            exported_program = torch.export.export(
                prediction_head,
                (example_embeddings,),
                dynamic_shapes=dynamic_shapes,
            )

            head_path = os.path.join(output_dir, f"head_{head_name}_{dtype}.pt2")
            torch.export.save(exported_program, head_path)
            print(f"Exported to {head_path}")
            exported_heads.append(head_name)

    print(f"Heads exported: {', '.join(exported_heads)}")


def main():
    parser = argparse.ArgumentParser(
        description="Export optimized model components using torch.export"
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for exported models (default: checkpoint_dir/exported)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="both",
        choices=["float32", "float16", "both"],
        help="Data type for export (default: float32). Use float16 for GPU inference, float32 for CPU, or both for both versions.",
    )
    parser.add_argument(
        "--backbone-only", action="store_true", help="Export only the backbone"
    )
    parser.add_argument(
        "--heads-only", action="store_true", help="Export only the prediction heads"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        raise ValueError(f"Checkpoint directory not found: {args.checkpoint}")

    output_dir = args.output_dir or os.path.join(args.checkpoint, "exported")
    os.makedirs(output_dir, exist_ok=True)

    config = load_config(args.checkpoint)

    config_src = os.path.join(args.checkpoint, "config.yaml")
    config_dst = os.path.join(output_dir, "config.yaml")
    shutil.copy2(config_src, config_dst)
    print(f"Config copied to {config_dst}")

    export_backbone_flag = not args.heads_only
    export_heads_flag = not args.backbone_only

    dtypes_to_export = ["float32", "float16"] if args.dtype == "both" else [args.dtype]

    for dtype in dtypes_to_export:
        if len(dtypes_to_export) > 1:
            print(f"Exporting {dtype} version")

        if export_backbone_flag:
            export_backbone(
                args.checkpoint,
                config,
                output_dir,
                dtype=dtype,
            )

        if export_heads_flag:
            export_heads(
                args.checkpoint,
                config,
                output_dir,
                dtype=dtype,
            )

    print(f"Export done: Models saved to {output_dir}")


if __name__ == "__main__":
    main()
