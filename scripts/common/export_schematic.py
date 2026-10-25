"""Export a saved dense MCNN artifact as a Minecraft schematic."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mcnn import (
    MCNNModel,
    model_from_artifact_state,
    save_schematic,
    validate_export_signals,
)


def load_model(path: Path) -> MCNNModel:
    artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise ValueError("artifact must contain a dictionary")
    state = artifact.get("model", artifact)
    if not isinstance(state, dict) or "layers" not in state:
        raise ValueError("artifact does not contain a dense MCNN model state")
    return model_from_artifact_state(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to a dense MCNN .pt artifact")
    parser.add_argument("--output", type=Path, help="Output .schem path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model
    if not model_path.exists():
        raise SystemExit(f"Model artifact not found: {model_path}")

    model = load_model(model_path)
    signal_check = validate_export_signals(model)
    if not bool(signal_check["ok"]):
        raise SystemExit(
            "model is not physically exportable: "
            f"violations={signal_check['violations']} max_signal={signal_check['max_signal']}"
        )

    output = args.output or model_path.with_suffix(".schem")
    saved = save_schematic(model, output)
    print(f"saved schematic={saved}")
    print(
        "signal_check="
        f"violations={signal_check['violations']} "
        f"max_signal={signal_check['max_signal']} "
        f"min_signal={signal_check['min_signal']}"
    )


if __name__ == "__main__":
    main()
