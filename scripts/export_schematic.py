"""Regenerate model.schem from a saved MCNN model.pt artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mcnn import model_from_artifact_state, save_schematic, validate_export_signals


DEFAULT_RUN_DIR = Path("artifacts/shift_digits_64_64_10")


def load_model(path: Path):
    artifact = torch.load(path, map_location="cpu")
    return model_from_artifact_state(artifact["model"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.run_dir / "model.pt"
    if not model_path.exists():
        raise SystemExit(f"Model artifact not found: {model_path}")

    model = load_model(model_path)
    signal_check = validate_export_signals(model)
    if not bool(signal_check["ok"]):
        raise SystemExit(
            "model is not physically exportable: "
            f"violations={signal_check['violations']} max_signal={signal_check['max_signal']}"
        )

    saved = save_schematic(model, args.run_dir / "model")
    print(f"saved schematic={saved}")
    print(
        "signal_check="
        f"violations={signal_check['violations']} "
        f"max_signal={signal_check['max_signal']} "
        f"min_signal={signal_check['min_signal']}"
    )


if __name__ == "__main__":
    main()
