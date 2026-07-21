from __future__ import annotations
from pathlib import Path
import mcschematic

from .discrete import DiscreteMCNN
from .constants import BIAS_SOURCE_SLOTS, GLITCHED_UNSTACKABLE_STACK_LIMIT, MAX_BIAS_SIGNAL_STRENGTH


# ---------------------------------------------------------------------------
# Block palette
# ---------------------------------------------------------------------------

SUPPORT = "minecraft:light_gray_concrete"
COMPARATOR_WEST = "minecraft:comparator[facing=west,mode=subtract,powered=false]"
COMPARATOR_SOUTH = "minecraft:comparator[facing=south,mode=subtract,powered=false]"
REDSTONE_BLOCK = "minecraft:redstone_block"
TORCH_EAST = "minecraft:redstone_wall_torch[facing=east,lit=true]"


def _barrel(signal: int) -> str:
    """Barrel block entity with the given comparator signal strength."""
    if signal < 0 or signal > MAX_BIAS_SIGNAL_STRENGTH:
        raise ValueError(f"barrel signal {signal} out of range [0, {MAX_BIAS_SIGNAL_STRENGTH}]")
    if signal == 0:
        return 'minecraft:barrel{CustomName:\'{"italic":false,"text":"0"}\'}'
    items = _glitched_unstackable_items(signal, BIAS_SOURCE_SLOTS)
    if items is None:
        raise ValueError(f"cannot encode barrel signal {signal}")
    return f'minecraft:barrel{{CustomName:\'{{"italic":false,"text":"{signal}"}}\',Items:[{",".join(items)}]}}'


def _glitched_unstackable_items(signal: int, slot_count: int) -> list[str] | None:
    import math
    for bucket_total in range(1, slot_count * GLITCHED_UNSTACKABLE_STACK_LIMIT + 1):
        sig = math.floor(1 + 14 * bucket_total / slot_count)
        if sig == signal:
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
    _set = schematic.setBlock
    _set((x + 0, y + 0, z + 0), SUPPORT)
    _set((x + 0, y + 1, z + 0), COMPARATOR_WEST)
    _set((x + 1, y + 1, z + 0), SUPPORT)
    _set((x + 0, y + 0, z + 1), REDSTONE_BLOCK)
    _set((x + 0, y + 1, z + 1), COMPARATOR_SOUTH)
    _set((x + 1, y + 1, z + 1), REDSTONE_BLOCK)
    _set((x + 0, y + 1, z + 2), SUPPORT)
    _set((x + 1, y + 1, z + 2), TORCH_EAST)
    _set((x + 0, y + 1, z + 3), _barrel(barrel_signal))


def _place_barrel(schematic: mcschematic.MCSchematic, x: int, y: int, z: int, signal: int):
    """Place the bias barrel at the start of a neuron row."""
    schematic.setBlock((x, y, z), _barrel(signal))


# ---------------------------------------------------------------------------
# Build schematic
# ---------------------------------------------------------------------------

def build_schematic(model: DiscreteMCNN) -> mcschematic.MCSchematic:
    """Tile nn_modules for each weight in a diagonal grid.

    The module places redstone blocks at (x+0, y+0, z+1) and (x+1, y+1, z+1).
    For a checkerboard in the (x, y, z=1) plane, we need alternating parity.

    Module origin:
        x = layer_origin_x + output_index + input_index * 2
        y = layer_origin_y + output_index
    """
    schematic = mcschematic.MCSchematic()

    next_layer_x = 0

    for layer in model.layers:
        input_size = int(layer.weight.shape[1])
        output_size = int(layer.weight.shape[0])


        for output_index in range(output_size):
            bias = int(layer.bias[output_index].item())
            for input_index in range(input_size):
                weight = int(layer.weight[output_index, input_index].item())

                barrel_signal = 15 + abs(weight)
                if output_index % 2 == 0:
                    bias += max(0,weight)
                else:
                    bias -= min(weight,0)

                mx = next_layer_x + output_index + input_index * 2
                my = output_index
                _place_module(schematic, mx, my, 0, barrel_signal)

            _place_barrel(schematic, next_layer_x+output_index-1, output_index+1, 0,bias)
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
