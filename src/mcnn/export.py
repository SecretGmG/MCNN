from __future__ import annotations

from pathlib import Path

import mcschematic

from .constants import BIAS_SOURCE_SLOTS, GLITCHED_UNSTACKABLE_STACK_LIMIT, MAX_BIAS_SIGNAL_STRENGTH
from .discrete import DiscreteMCNN

# ---------------------------------------------------------------------------
# Block helpers
# ---------------------------------------------------------------------------


def comparator(facing: str, mode: str = "subtract") -> str:
    return f"minecraft:comparator[facing={facing},mode={mode},powered=false]"


def wall_torch(facing: str) -> str:
    return f"minecraft:redstone_wall_torch[facing={facing},lit=true]"


def repeater(facing: str, delay: int = 1) -> str:
    return f"minecraft:repeater[delay={delay},facing={facing},locked=false,powered=false]"


# ---------------------------------------------------------------------------
# Block palette
# ---------------------------------------------------------------------------

SUPPORT_WEIGHT = "minecraft:light_gray_concrete"
SUPPORT_IN = "minecraft:light_blue_concrete"
SUPPORT_OUT = "minecraft:blue_concrete"
SUPPORT_CONNECT = "minecraft:purple_concrete"
REDSTONE_BLOCK = "minecraft:redstone_block"
TORCH = "minecraft:redstone_torch[lit=true]"
WIRE = "minecraft:redstone_wire"
SLAB = "minecraft:smooth_stone_slab[type=top]"


# ---------------------------------------------------------------------------
# Barrel signal encoding
# ---------------------------------------------------------------------------


def _barrel(signal: int) -> str:
    """Barrel block entity with the given comparator signal strength."""
    if signal < 0 or signal > MAX_BIAS_SIGNAL_STRENGTH:
        raise ValueError(f"barrel signal {signal} out of range [0, {MAX_BIAS_SIGNAL_STRENGTH}]")

    if signal == 0:
        return 'minecraft:barrel{CustomName:\'{"italic":false,"text":"0"}\'}'

    items = _glitched_unstackable_items(signal, BIAS_SOURCE_SLOTS)
    if items is None:
        raise ValueError(f"cannot encode barrel signal {signal}")

    return (
        "minecraft:barrel{"
        f'CustomName:\'{{"italic":false,"text":"{signal}"}}\','
        f"Items:[{','.join(items)}]"
        "}"
    )


def _glitched_unstackable_items(signal: int, slot_count: int) -> list[str] | None:
    import math

    for bucket_total in range(1, slot_count * GLITCHED_UNSTACKABLE_STACK_LIMIT + 1):
        sig = math.floor(1 + 14 * bucket_total / slot_count)
        if sig != signal:
            continue

        items = []
        remaining = bucket_total
        slot = 0

        while remaining:
            count = min(GLITCHED_UNSTACKABLE_STACK_LIMIT, remaining)
            items.append(f'{{Slot:{slot}b,id:"minecraft:water_bucket",count:{count}}}')
            remaining -= count
            slot += 1

        return items

    return None


# ---------------------------------------------------------------------------
# Module placement
# ---------------------------------------------------------------------------


def _place_module(schematic: mcschematic.MCSchematic, x: int, y: int, z: int, barrel_signal: int):
    """Place one weight module at the given origin.

    If barrel_signal is None or 0, the barrel is omitted (zero weight).
    """
    set_block = schematic.setBlock

    set_block((x + 0, y + 0, z + 0), SUPPORT_WEIGHT)
    set_block((x + 0, y + 1, z + 0), comparator("west"))
    set_block((x + 1, y + 1, z + 0), SUPPORT_WEIGHT)

    set_block((x + 0, y + 0, z + 1), SUPPORT_WEIGHT)
    set_block((x + 0, y + 1, z + 1), comparator("south"))
    set_block((x + 1, y + 1, z + 1), SUPPORT_WEIGHT)

    set_block((x + 0, y + 1, z + 2), SUPPORT_IN)
    set_block((x + 1, y + 1, z + 2), wall_torch("east"))

    set_block((x + 0, y + 1, z + 3), _barrel(barrel_signal))


def _place_barrel(schematic: mcschematic.MCSchematic, x: int, y: int, z: int, signal: int):
    """Place the bias barrel at the start of a neuron row."""
    schematic.setBlock((x, y, z), _barrel(signal))


def _build_wire_out(schematic: mcschematic.MCSchematic, x: int, y: int, z: int) -> int:
    set_block = schematic.setBlock

    set_block((x - 1, y, z - 1), SUPPORT_OUT)
    set_block((x - 1, y + 1, z - 1), repeater("south"))

    if y % 2 == 1:
        set_block((x - 1, y + 1, z - 2), SUPPORT_OUT)

    wire_z = z - 2 - (y % 2)

    flip = False
    delay = 0

    for j in range(y + 2):
        wire_y = y - j

        if j % 15 == 14:
            set_block((x - 1 + j, wire_y + 1, wire_z), wall_torch("east"))
            set_block((x + j, wire_y + 1, wire_z), SUPPORT_OUT)

            flip = not flip
            wire_z = z - 2 - ((y + flip) % 2)
            delay += 1

        set_block((x - 1 + j, wire_y, wire_z), SUPPORT_OUT)
        set_block((x - 1 + j, wire_y + 1, wire_z), WIRE)

    if not flip:
        set_block((x + y, -2, wire_z - 1), SUPPORT_OUT)
        set_block((x + y, -1, wire_z - 1), repeater("south"))
    else:
        set_block((x + y, -1, wire_z - 1), wall_torch("north"))

    if z - 2 == wire_z:
        set_block((x + y, -2, z - 4), SUPPORT_OUT)
        set_block((x + y, -1, z - 4), WIRE)

    set_block((x+y,-2,z-5), SLAB)
    set_block((x+y,-1,z-5), repeater("south"))


    return delay


def _build_connector(schematic: mcschematic.MCSchematic, x: int, y: int, z: int, delay: int):
    set_block = schematic.setBlock

    set_block((x, y - 3, z + 2), SUPPORT_CONNECT)
    set_block((x, y - 2, z + 2), TORCH)
    set_block((x, y - 1, z + 2), SUPPORT_CONNECT)
    set_block((x, y - 0, z + 2), TORCH)

    for i in reversed(range(-1, 6)):
        set_block((x, y - 4, z - i), SUPPORT_CONNECT)

        if delay >= 2:
            set_block((x + y, y - 3, z - i), repeater("north", 2))
            delay -= 2
        elif delay == 1:
            set_block((x + y, y - 3, z - i), repeater("north"))
            delay = 0
        else:
            set_block((x, y - 3, z - i), WIRE)

    connector_z = z - 6
    place_top_block = True

    while delay > 0:
        set_block((x, y - 4, connector_z), SUPPORT_CONNECT)
        set_block((x, y - 2, connector_z), SUPPORT_CONNECT)

        if delay >= 2:
            schematic.setBlock((x + y, y - 3, connector_z), repeater("north", 2))
            return delay - 2

        if delay == 1:
            schematic.setBlock((x + y, y - 3, connector_z), repeater("north"))
            return 0

        if delay >= 2:
            set_block((x + y, y - 1, connector_z), repeater("south", 2))
            delay -= 2
        elif delay == 1:
            set_block((x + y, y - 1, connector_z), repeater("south"))
            delay = 0
        else:
            set_block((x, y - 1, connector_z), WIRE)
            place_top_block = False

        connector_z -= 1

    if place_top_block:
        set_block((x, y - 1, connector_z), SUPPORT_CONNECT)

    set_block((x, y - 2, connector_z), WIRE)
    set_block((x, y - 3, connector_z), SUPPORT_CONNECT)


# ---------------------------------------------------------------------------
# Build schematic
# ---------------------------------------------------------------------------


def build_schematic(model: DiscreteMCNN) -> mcschematic.MCSchematic:
    """Tile nn_modules for each weight in a diagonal grid.

    The module places redstone blocks at (x+0, y+0, z+1) and
    (x+1, y+1, z+1). For a checkerboard in the (x, y, z=1) plane,
    we need alternating parity.

    Module origin:
        x = layer_origin_x + output_index + input_index * 2
        y = layer_origin_y + output_index
    """
    schematic = mcschematic.MCSchematic()
    next_layer_x = 0

    for layer_index, layer in enumerate(model.layers):
        input_size = int(layer.weight.shape[1])
        output_size = int(layer.weight.shape[0])

        delays = []

        for output_index in range(output_size):
            bias_signal = int(layer.bias[output_index].item())

            for input_index in range(input_size):
                weight = int(layer.weight[output_index, input_index].item())

                flip = output_index % 2 == 0

                barrel_signal = 15 + weight * (1 if flip else -1)

                if flip:
                    bias_signal += barrel_signal
                else:
                    bias_signal += 15

                module_x = next_layer_x + output_index + input_index * 2
                module_y = output_index

                _place_module(schematic, module_x, module_y, 0, barrel_signal)

            _place_barrel(schematic, next_layer_x + output_index - 1, output_index + 1, 0, bias_signal)

            if layer_index != len(model.layers) - 1:
                delay = _build_wire_out(schematic, next_layer_x + output_index + input_size * 2, output_index, 0)
                delays.append(delay)

        if layer_index != len(model.layers) - 1:
            max_delay = max(delays)

            for output_index in range(output_size):
                pass
                _build_connector(
                    schematic, next_layer_x + input_size * 2 + output_index * 2, 0, 0, max_delay - delays[output_index]
                )

        next_layer_x += input_size * 2

    return schematic


def save_schematic(
    model: DiscreteMCNN,
    output_path: str | Path,
    *,
    version: mcschematic.Version = mcschematic.Version.JE_1_21_5,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schematic = build_schematic(model)
    schematic.save(str(output_path.parent), output_path.stem, version)

    return output_path.with_suffix(".schem")
