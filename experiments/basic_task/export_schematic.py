"""Regenerate a Minecraft schematic from a saved MCNN model artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mcnn import model_from_artifact_state, save_schematic, validate_export_signals


def load_model(path: Path):
    artifact = torch.load(path, map_location="cpu")
    return model_from_artifact_state(artifact["model"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/run/model.pt"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path without or with .schem suffix. Defaults to model path stem.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model artifact not found: {args.model}")

    output = args.output
    if output is None:
        output = args.model.with_suffix("")

    model = load_model(args.model)
    signal_check = validate_export_signals(model)
    if not bool(signal_check["ok"]):
        raise SystemExit(
            "model is not physically exportable: "
            f"violations={signal_check['violations']} max_signal={signal_check['max_signal']}"
        )

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
