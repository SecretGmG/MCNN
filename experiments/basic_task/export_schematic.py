"""Export a saved discrete MCNN artifact to a staggered-layer schematic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcnn import SchematicExportConfig, discrete_from_state_dict, export_plan_from_discrete, save_module_atlas, save_schematic  # noqa: E402


def export(args: argparse.Namespace) -> None:
    if args.module_atlas:
        saved_path = save_module_atlas(args.output)
        print(f"saved_module_atlas={saved_path}")
        return

    artifact = torch.load(args.artifact, map_location="cpu")
    model = discrete_from_state_dict(artifact["model"])
    plan = export_plan_from_discrete(model)
    config = SchematicExportConfig(
        emit_weights=args.emit_weights,
        emit_subtraction=args.emit_subtraction,
        emit_input_wiring=args.emit_input_wiring,
        emit_output_wiring=args.emit_output_wiring,
        emit_layer_io_connectors=args.emit_layer_io_connectors,
    )
    saved_path = save_schematic(plan, args.output, config=config)
    print(f"saved_schematic={saved_path}")
    print(f"layer_sizes={plan.layer_sizes}")
    for layer in plan.layers:
        connection_count = sum(len(neuron.connections) for neuron in layer.neurons)
        max_physical_bias = max((neuron.physical_bias for neuron in layer.neurons), default=0)
        print(
            f"layer={layer.layer_index} input={layer.input_size} output={layer.output_size} "
            f"connections={connection_count} max_physical_bias={max_physical_bias}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default=ROOT / "artifacts" / "mcnn_digits_8x8_structural.pt")
    parser.add_argument("--output", default=ROOT / "artifacts" / "mcnn_digits_8x8_layers.schem")
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
