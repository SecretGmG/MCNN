"""Export a saved discrete MCNN artifact to a staggered-layer schematic."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

from mcnn import (
    SchematicExportConfig,
    comparator_subtractions,
    discrete_from_state_dict,
    save_module_atlas,
    save_schematic,
)


def export(args: argparse.Namespace) -> None:
    if args.module_atlas:
        saved_path = save_module_atlas(args.output)
        print(f"saved_module_atlas={saved_path}")
        return

    artifact = torch.load(args.artifact, map_location="cpu")
    model = discrete_from_state_dict(artifact["model"])
    config = SchematicExportConfig(
        emit_weights=args.emit_weights,
        emit_subtraction=args.emit_subtraction,
        emit_input_wiring=args.emit_input_wiring,
        emit_output_wiring=args.emit_output_wiring,
        emit_layer_io_connectors=args.emit_layer_io_connectors,
    )
    saved_path = save_schematic(model, args.output, config=config)
    print(f"saved_schematic={saved_path}")
    print(f"layer_sizes={model.layer_sizes()}")
    for layer_index, layer in enumerate(model.layers):
        connection_count = int(layer.weight.count_nonzero().item())
        max_physical_bias = max(
            (
                int(layer.bias[output_index].item())
                + sum(
                    comparator_subtractions(
                        int(layer.weight[output_index, input_index].item())
                    ).inactive
                    for input_index in range(layer.weight.shape[1])
                )
                for output_index in range(layer.weight.shape[0])
            ),
            default=0,
        )
        print(
            f"layer={layer_index} input={layer.weight.shape[1]} output={layer.weight.shape[0]} "
            f"connections={connection_count} max_physical_bias={max_physical_bias}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact", default=ROOT / "artifacts" / "mcnn_digits_8x8_structural.pt"
    )
    parser.add_argument(
        "--output", default=ROOT / "artifacts" / "mcnn_digits_8x8_layers.schem"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--weights-only",
        action="store_true",
        help="export only furnace-backed weight and bias signal sources",
    )
    mode.add_argument(
        "--subtraction-only",
        action="store_true",
        help="export only the subtraction mechanism with purple/magenta scaffold blocks",
    )
    mode.add_argument(
        "--input-wiring-only",
        action="store_true",
        help="export only the input wiring mechanism",
    )
    mode.add_argument(
        "--output-wiring-only",
        action="store_true",
        help="export only the output wiring mechanism",
    )
    mode.add_argument(
        "--module-atlas",
        action="store_true",
        help="export isolated output-wiring modules separated by air gaps",
    )
    parser.add_argument(
        "--include-input-wiring",
        action="store_true",
        help="add input wiring to the default weights plus subtraction export",
    )
    parser.add_argument(
        "--include-output-wiring",
        action="store_true",
        help="add output wiring to the default weights plus subtraction export",
    )
    parser.add_argument(
        "--include-layer-io-connectors",
        action="store_true",
        help="connect each layer's output wiring to the next layer's input wiring",
    )
    args = parser.parse_args()

    args.emit_weights = True
    args.emit_subtraction = True
    args.emit_input_wiring = args.include_input_wiring
    args.emit_output_wiring = args.include_output_wiring
    args.emit_layer_io_connectors = args.include_layer_io_connectors
    if args.weights_only:
        args.emit_weights = True
        args.emit_subtraction = False
        args.emit_input_wiring = False
        args.emit_output_wiring = False
        args.emit_layer_io_connectors = False
    elif args.subtraction_only:
        args.emit_weights = False
        args.emit_subtraction = True
        args.emit_input_wiring = False
        args.emit_output_wiring = False
        args.emit_layer_io_connectors = False
    elif args.input_wiring_only:
        args.emit_weights = False
        args.emit_subtraction = False
        args.emit_input_wiring = True
        args.emit_output_wiring = False
        args.emit_layer_io_connectors = False
    elif args.output_wiring_only:
        args.emit_weights = False
        args.emit_subtraction = False
        args.emit_input_wiring = False
        args.emit_output_wiring = True
        args.emit_layer_io_connectors = False
    return args


if __name__ == "__main__":
    export(parse_args())
