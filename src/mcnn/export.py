"""Export planning for the Minecraft comparator-chain runtime.

This module converts an exact discrete MCNN model into a physical export plan,
then emits a schematic for the current dense comparator-chain layout.

Architecture summary
--------------------

Each neuron is a comparator subtraction chain fed by a large bias signal. Every
nonzero input connection is encoded as one repeated physical module:

- Mode A: inactive subtracts 15, active subtracts nonzero ``value``.
  Effective weight is ``15 - value``.
- Mode B: inactive subtracts nonzero ``value``, active subtracts 15.
  Effective weight is ``value - 15``.
- Mode C: zero effective weight. It is exported as a visible no-op slot.

Mode C matters for bias requirements: it contributes no default subtraction to
the physical chain. The exporter therefore tracks both the effective model bias
and the physical base signal required by the generated chain.

The effective model computes:

```text
y = clamp_int(sum_i(w_i * x_i) + b, output_min, output_max)
```

with the default output range clamped to ``0..15``. The physical chain computes
the same value by starting from a large signal and subtracting each emitted
module's inactive contribution. The physical base signal is:

```text
physical_bias = effective_bias + sum(inactive_subtraction for emitted modules)
```
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mcschematic

from .constants import (
    BIAS_SOURCE_SLOTS,
    GLITCHED_UNSTACKABLE_STACK_LIMIT,
    MAX_BIAS_SIGNAL_STRENGTH,
)
from .discrete import DiscreteMCNN
from .encoding import (
    ComparatorEncoding,
    comparator_subtractions,
    encode_comparator_weight,
)

BlockPlacement = tuple[int, int, int, str]

INPUT_PAIR_WIDTH = 4
INPUT_RELAY_PERIOD = 15
OUTPUT_LANE_FORWARD_Z = -2
OUTPUT_BLACK_LANE_FORWARD_Z = -3
OUTPUT_WHITE_LANE_FORWARD_Z = -2
MODULE_ATLAS_GAP = 6
MODULE_ATLAS_CORNER_BLOCK = "minecraft:stone"
FURNACE_SOURCE_SLOTS = 3
MAX_FURNACE_SIGNAL_STRENGTH = int(1 + 14 * GLITCHED_UNSTACKABLE_STACK_LIMIT)


def _wire(
    power: int = 0,
    *,
    east: str = "none",
    north: str = "none",
    south: str = "none",
    west: str = "none",
) -> str:
    return f"minecraft:redstone_wire[east={east},north={north},power={power},south={south},west={west}]"


@dataclass(frozen=True)
class SchematicExportConfig:
    """Geometry and block palette for the first schematic exporter.

    This is intentionally a dense debug layout. It places one horizontal
    perceptron row per output, with weight slots every second block. Consecutive
    outputs are placed one block higher and shifted one block along X. Neural
    layers are placed after each other along X so successor-layer bias
    source blocks occupy the previous layer's output-read positions.
    """

    input_pitch: int = 2
    layer_gap: int = 2
    emit_weights: bool = True
    emit_subtraction: bool = True
    emit_input_wiring: bool = False
    emit_output_wiring: bool = False
    emit_layer_io_connectors: bool = False
    target_block: str = "minecraft:target"
    support_block: str = "minecraft:light_gray_concrete"
    dust_cover_block: str = "minecraft:light_gray_concrete"
    input_wiring_white_block: str = "minecraft:white_concrete"
    input_wiring_black_block: str = "minecraft:black_concrete"
    placeholder_bias_block: str = "minecraft:magenta_concrete"
    placeholder_weight_block: str = "minecraft:purple_concrete"
    slab_block: str = "minecraft:smooth_stone_slab[type=top,waterlogged=false]"
    comparator_west: str = (
        "minecraft:comparator[facing=west,mode=subtract,powered=false]"
    )
    comparator_south_on: str = (
        "minecraft:comparator[facing=south,mode=compare,powered=true]"
    )
    comparator_south_off: str = (
        "minecraft:comparator[facing=south,mode=compare,powered=false]"
    )
    repeater_south: str = (
        "minecraft:repeater[delay=1,facing=south,locked=false,powered=false]"
    )
    input_repeater_south: str = (
        "minecraft:repeater[delay=1,facing=south,locked=false,powered=true]"
    )
    input_repeater_north: str = (
        "minecraft:repeater[delay=1,facing=north,locked=false,powered=true]"
    )
    output_repeater_south: str = (
        "minecraft:repeater[delay=1,facing=south,locked=false,powered=false]"
    )
    output_repeater_north: str = (
        "minecraft:repeater[delay=1,facing=north,locked=false,powered=false]"
    )
    torch_block: str = "minecraft:redstone_wall_torch[facing=north,lit=true]"


def signal_strength_furnace(signal_strength: int) -> str:
    """Return a furnace block entity that comparators read as ``signal_strength``."""

    if signal_strength < 0:
        raise ValueError(
            f"furnace signal strength {signal_strength} must be non-negative"
        )
    if signal_strength > MAX_FURNACE_SIGNAL_STRENGTH:
        raise ValueError(
            f"furnace signal strength {signal_strength} exceeds the configured "
            f"maximum {MAX_FURNACE_SIGNAL_STRENGTH}"
        )
    if signal_strength == 0:
        return """minecraft:furnace{CustomName:'{"italic":false,"text":"0"}'}"""

    items = furnace_signal_items(signal_strength)
    if items is None:
        raise ValueError(f"could not encode furnace signal strength {signal_strength}")

    return f"""minecraft:furnace{{CustomName:'{{"italic":false,"text":"{signal_strength}"}}',Items:[{",".join(items)}]}}"""


def signal_strength_barrel(signal_strength: int) -> str:
    """Return a barrel block entity that comparators read as ``signal_strength``."""

    if signal_strength < 0:
        raise ValueError(
            f"barrel signal strength {signal_strength} must be non-negative"
        )
    if signal_strength > MAX_BIAS_SIGNAL_STRENGTH:
        raise ValueError(
            f"barrel signal strength {signal_strength} exceeds the configured "
            f"maximum {MAX_BIAS_SIGNAL_STRENGTH}"
        )
    if signal_strength == 0:
        return """minecraft:barrel{CustomName:'{"italic":false,"text":"0"}'}"""

    items = _glitched_unstackable_items(signal_strength, BIAS_SOURCE_SLOTS)
    if items is None:
        raise ValueError(f"could not encode barrel signal strength {signal_strength}")

    return f"""minecraft:barrel{{CustomName:'{{"italic":false,"text":"{signal_strength}"}}',Items:[{",".join(items)}]}}"""


def is_representable_bias_signal(signal_strength: int) -> bool:
    return 0 <= signal_strength <= MAX_BIAS_SIGNAL_STRENGTH and (
        signal_strength == 0
        or _glitched_unstackable_items(signal_strength, BIAS_SOURCE_SLOTS) is not None
    )


def furnace_signal_items(signal_strength: int) -> list[str] | None:
    if signal_strength < 0 or signal_strength > MAX_FURNACE_SIGNAL_STRENGTH:
        return None
    if signal_strength == 0:
        return []
    if signal_strength <= 15:
        return _stackable_furnace_items(signal_strength)
    return _glitched_unstackable_items(signal_strength, FURNACE_SOURCE_SLOTS)


def _stackable_furnace_items(signal_strength: int) -> list[str]:
    total_items = math.ceil((signal_strength - 1) * 3 * 64 / 14)
    items = []
    for slot in range(3):
        count = min(64, total_items - slot * 64)
        if count <= 0:
            break
        items.append(f'{{Slot:{slot}b,id:"minecraft:redstone",count:{count}}}')
    return items


def _glitched_unstackable_items(
    signal_strength: int, slot_count: int
) -> list[str] | None:
    for bucket_total in range(1, slot_count * GLITCHED_UNSTACKABLE_STACK_LIMIT + 1):
        signal = math.floor(1 + 14 * bucket_total / slot_count)
        if signal == signal_strength:
            items = []
            remaining_buckets = bucket_total
            slot = 0
            while remaining_buckets:
                count = min(GLITCHED_UNSTACKABLE_STACK_LIMIT, remaining_buckets)
                items.append(
                    f'{{Slot:{slot}b,id:"minecraft:water_bucket",count:{count}}}'
                )
                remaining_buckets -= count
                slot += 1
            return items
    return None


def _set_safe(
    schematic: mcschematic.MCSchematic, position: tuple[int, int, int], block: str
) -> None:
    schematic.setBlock(position, block)


def _place_relative(
    schematic: mcschematic.MCSchematic,
    origin: tuple[int, int, int],
    placement: BlockPlacement,
) -> None:
    dx, dy, dz, block = placement
    _set_safe(schematic, (origin[0] + dx, origin[1] + dy, origin[2] + dz), block)


def _place_pattern(
    schematic: mcschematic.MCSchematic,
    origin: tuple[int, int, int],
    placements: tuple[BlockPlacement, ...],
) -> None:
    for placement in placements:
        _place_relative(schematic, origin, placement)


def _place_if_air(
    schematic: mcschematic.MCSchematic,
    origin: tuple[int, int, int],
    placement: BlockPlacement,
) -> None:
    dx, dy, dz, block = placement
    position = (origin[0] + dx, origin[1] + dy, origin[2] + dz)
    if schematic.getBlockStateAt(position) == "minecraft:air":
        _set_safe(schematic, position, block)


def _set_support(
    schematic: mcschematic.MCSchematic,
    position: tuple[int, int, int],
    config: SchematicExportConfig,
) -> None:
    if schematic.getBlockStateAt(position) in ("minecraft:air", config.slab_block):
        _set_safe(schematic, position, config.support_block)


def _place_repeater(
    schematic: mcschematic.MCSchematic,
    position: tuple[int, int, int],
    repeater: str,
    config: SchematicExportConfig,
    *,
    support_block: str | None = None,
) -> None:
    support_position = (position[0], position[1] - 1, position[2])
    if support_block is None:
        _set_support(schematic, support_position, config)
    elif schematic.getBlockStateAt(support_position) == "minecraft:air":
        _set_safe(schematic, support_position, support_block)
    _set_safe(schematic, position, repeater)


def _input_pair_bases(x: int, input_size: int) -> tuple[tuple[int, bool], ...]:
    return tuple(
        (x + pair_index * INPUT_PAIR_WIDTH, pair_index * 2 + 1 < input_size)
        for pair_index in range((input_size + 1) // 2)
    )


def _is_black_lane(lane_index: int, lane_phase: int) -> bool:
    return (lane_index + lane_phase) % 2 == 0


def _lane_block(config: SchematicExportConfig, lane_index: int, lane_phase: int) -> str:
    return (
        config.input_wiring_black_block
        if _is_black_lane(lane_index, lane_phase)
        else config.input_wiring_white_block
    )


def _layer_placements(
    model: DiscreteMCNN, config: SchematicExportConfig
) -> list[tuple[int, int, int, int]]:
    """Return (input_size, origin_x, origin_y, input_lane_phase) per layer."""
    placements: list[tuple[int, int, int, int]] = []
    next_layer_x = 0
    next_lane_phase = 0
    for layer in model.layers:
        input_size = int(layer.weight.shape[1])
        placements.append((input_size, next_layer_x, 0, next_lane_phase))
        next_layer_x += input_size * config.input_pitch
        next_lane_phase = (next_lane_phase + input_size) % 2
    return placements


def _place_input_targets(
    schematic: mcschematic.MCSchematic,
    base_x: int,
    y: int,
    z: int,
    has_second_target: bool,
) -> None:
    _set_safe(schematic, (base_x, y, z), "minecraft:target[power=0]")
    if has_second_target:
        _set_safe(schematic, (base_x + 2, y, z), "minecraft:target[power=0]")


def _place_furnace_top_dust(
    schematic: mcschematic.MCSchematic,
    position: tuple[int, int, int],
    dust: str,
    config: SchematicExportConfig,
) -> None:
    _set_safe(schematic, position, dust)
    _set_safe(
        schematic, (position[0], position[1] + 1, position[2]), config.dust_cover_block
    )


def _place_bias_source(
    schematic: mcschematic.MCSchematic,
    position: tuple[int, int, int],
    signal_strength: int,
    config: SchematicExportConfig,
) -> None:
    source_block = (
        signal_strength_barrel(signal_strength)
        if config.emit_weights
        else config.placeholder_bias_block
    )
    if config.emit_weights or config.emit_subtraction:
        _set_safe(schematic, position, source_block)


def _place_weight_source(
    schematic: mcschematic.MCSchematic,
    position: tuple[int, int, int],
    control_position: tuple[int, int, int],
    signal_strength: int,
    signal_dust: str,
    config: SchematicExportConfig,
) -> None:
    if not (config.emit_weights or config.emit_subtraction):
        return

    source_block = (
        signal_strength_furnace(signal_strength)
        if config.emit_weights
        else config.placeholder_weight_block
    )
    _set_safe(schematic, position, config.support_block)
    _set_safe(schematic, control_position, source_block)
    if config.emit_subtraction:
        _place_furnace_top_dust(
            schematic, (position[0], position[1] + 1, position[2]), signal_dust, config
        )


def _place_connection_module(
    schematic: mcschematic.MCSchematic,
    x: int,
    y: int,
    z: int,
    mode: str,
    value: int,
    has_next_input: bool,
    config: SchematicExportConfig,
) -> None:
    """Place one nonzero connection module.

    The dense geometry keeps the visible weight matrix compact:

    - each input owns a two-block slot along X,
    - purple block marks the stored ``0..15`` weight signal,
    - target block marks that input's external signal position,
    - comparator/control blocks sit between the chain and target-side signal,
    - torch means Mode A and repeater means Mode B,
    - Mode C keeps the purple/target slot visible but emits no active control.
    """

    column_x = x + 1

    if config.emit_subtraction:
        _set_support(schematic, (column_x, y - 1, z), config)
        _set_safe(schematic, (column_x, y, z), config.comparator_west)
        if has_next_input:
            _set_support(schematic, (column_x + 1, y, z), config)

    if mode == "C":
        return

    signal_dust = (
        _wire(power=15, north="side", south="side")
        if mode == "A"
        else _wire(north="side", south="side")
    )
    _place_weight_source(
        schematic,
        (column_x, y, z + 2),
        (column_x, y, z + 3),
        value,
        signal_dust,
        config,
    )

    if not config.emit_subtraction:
        return

    is_mode_a = mode == "A"
    control_comparator = (
        config.comparator_south_on if is_mode_a else config.comparator_south_off
    )
    control_block = config.torch_block if is_mode_a else config.repeater_south
    _set_safe(schematic, (column_x, y - 1, z + 1), config.support_block)
    _set_safe(schematic, (column_x, y, z + 1), control_comparator)
    if not is_mode_a:
        _set_support(schematic, (column_x, y, z + 3), config)
        _place_repeater(schematic, (column_x, y + 1, z + 3), control_block, config)
    else:
        _set_safe(schematic, (column_x, y + 1, z + 3), control_block)
    _set_safe(schematic, (column_x, y + 1, z + 4), "minecraft:target[power=0]")


def _place_single_neuron_output_wiring(
    schematic: mcschematic.MCSchematic,
    x: int,
    y: int,
    z: int,
    input_size: int,
    config: SchematicExportConfig,
) -> None:
    if input_size != 3 or not config.emit_subtraction:
        return

    _place_pattern(
        schematic,
        (x, y, z),
        (
            (0, -1, 5, config.support_block),
            (3, -1, 5, _wire(west="side", east="up")),
            (4, -1, 5, config.support_block),
            (1, -1, 6, _wire(west="side", east="up")),
            (2, -1, 6, config.support_block),
            (0, 0, 5, _wire(west="side", east="up")),
            (1, 0, 5, config.support_block),
            (3, 0, 5, config.slab_block),
            (4, 0, 5, _wire(west="side", east="up")),
            (5, 0, 5, config.support_block),
            (2, 0, 6, _wire(west="side", east="up")),
            (3, 0, 6, config.support_block),
            (1, 1, 5, _wire(east="up", north="side", west="side")),
            (2, 1, 5, config.support_block),
            (5, 1, 5, _wire(west="side", east="up")),
            (3, 1, 6, _wire(east="up", north="side", west="side")),
            (4, 1, 6, config.support_block),
        ),
    )
    _place_repeater(schematic, (x + 3, y + 1, z + 5), config.repeater_south, config)


def _place_neuron(
    schematic: mcschematic.MCSchematic,
    layer_origin_x: int,
    layer_origin_y: int,
    layer_origin_z: int,
    output_index: int,
    physical_bias: int,
    connections: list[tuple[int, str, int]],
    config: SchematicExportConfig,
) -> None:
    """Place one neuron row.

    ``connections`` is a list of ``(input_index, mode, value)`` tuples.
    """
    row_x, row_y, row_z = _neuron_bias_position(
        layer_origin_x, layer_origin_y, layer_origin_z, output_index
    )

    _place_bias_source(schematic, (row_x, row_y, row_z), physical_bias, config)

    for input_index, mode, value in connections:
        x = row_x + input_index * config.input_pitch
        has_next_input = input_index < len(connections) - 1
        _place_connection_module(
            schematic, x, row_y, row_z, mode, value, has_next_input, config
        )

    _place_single_neuron_output_wiring(
        schematic, row_x, row_y, row_z, len(connections), config
    )


def _neuron_bias_position(
    layer_origin_x: int,
    layer_origin_y: int,
    layer_origin_z: int,
    output_index: int,
) -> tuple[int, int, int]:
    return (
        layer_origin_x + output_index,
        layer_origin_y + output_index + 1,
        layer_origin_z,
    )


def _place_normal_input_wiring_row(
    schematic: mcschematic.MCSchematic,
    x: int,
    y: int,
    z: int,
    signal_power: int,
    input_size: int,
    config: SchematicExportConfig,
    *,
    lane_phase: int = 0,
    adds_extender_step: bool = False,
    is_top_row: bool = False,
    is_bottom_row: bool = False,
) -> None:
    for input_index in range(input_size):
        target_x = x + input_index * config.input_pitch
        uses_black_lane = _is_black_lane(input_index, lane_phase)
        west = "none" if signal_power == 15 else "side"

        _set_safe(schematic, (target_x, y, z), "minecraft:target[power=0]")
        if uses_black_lane:
            _place_repeater(
                schematic,
                (target_x, y, z + 1),
                config.input_repeater_south,
                config,
                support_block=config.slab_block,
            )
            if not is_top_row:
                _set_safe(schematic, (target_x + 1, y, z + 1), config.slab_block)
            if not is_bottom_row:
                _set_safe(
                    schematic,
                    (target_x, y, z + 2),
                    _wire(
                        power=signal_power,
                        east="up",
                        north="side",
                        south="none",
                        west=west,
                    ),
                )
            if not is_top_row:
                _set_safe(
                    schematic, (target_x + 1, y, z + 2), config.input_wiring_black_block
                )
        else:
            if not is_bottom_row:
                _set_safe(
                    schematic,
                    (target_x, y, z + 1),
                    _wire(
                        power=signal_power,
                        east="up",
                        north="side",
                        south="none",
                        west=west,
                    ),
                )
            if not is_top_row:
                _set_safe(
                    schematic, (target_x + 1, y, z + 1), config.input_wiring_white_block
                )

        if adds_extender_step and not is_top_row:
            if uses_black_lane:
                _set_safe(
                    schematic, (target_x + 1, y, z + 3), config.input_wiring_black_block
                )
                _set_safe(
                    schematic, (target_x + 1, y, z + 4), config.input_wiring_black_block
                )
            else:
                _set_safe(schematic, (target_x + 1, y, z + 2), config.slab_block)
                _set_safe(
                    schematic, (target_x + 1, y, z + 3), config.input_wiring_white_block
                )


def _place_input_wiring_extender_lower_row(
    schematic: mcschematic.MCSchematic,
    x: int,
    y: int,
    z: int,
    input_size: int,
    config: SchematicExportConfig,
    lane_phase: int,
) -> None:
    for pair_index, (base_x, has_second_target) in enumerate(
        _input_pair_bases(x, input_size)
    ):
        first_lane = pair_index * 2
        second_lane = first_lane + 1
        _place_input_targets(schematic, base_x, y, z, has_second_target)
        _place_repeater(
            schematic,
            (base_x, y, z + 1),
            config.input_repeater_south,
            config,
            support_block=config.slab_block,
        )
        _set_safe(schematic, (base_x + 1, y, z + 1), config.slab_block)
        _set_safe(
            schematic,
            (base_x + 2, y, z + 1),
            _wire(power=1, east="none", north="side", south="side", west="side"),
        )
        _set_safe(
            schematic,
            (base_x + 3, y, z + 1),
            _lane_block(config, second_lane, lane_phase),
        )
        _set_safe(
            schematic,
            (base_x, y, z + 2),
            _wire(power=1, east="none", north="side", south="side", west="side"),
        )
        _set_safe(
            schematic,
            (base_x + 1, y, z + 2),
            _lane_block(config, first_lane, lane_phase),
        )
        _place_repeater(
            schematic,
            (base_x + 2, y, z + 2),
            config.input_repeater_north,
            config,
            support_block=config.slab_block,
        )
        _set_safe(schematic, (base_x + 3, y, z + 2), config.slab_block)
        _place_repeater(
            schematic,
            (base_x, y, z + 3),
            config.input_repeater_north,
            config,
            support_block=config.slab_block,
        )
        _set_safe(
            schematic,
            (base_x + 1, y, z + 3),
            _lane_block(config, first_lane, lane_phase),
        )
        _set_safe(
            schematic,
            (base_x + 2, y, z + 3),
            _wire(power=15, east="up", north="side", west="none"),
        )
        _set_safe(
            schematic,
            (base_x + 3, y, z + 3),
            _lane_block(config, second_lane, lane_phase),
        )
        _set_safe(
            schematic,
            (base_x, y, z + 4),
            _wire(power=15, east="up", north="side", west="none"),
        )
        _set_safe(
            schematic,
            (base_x + 1, y, z + 4),
            _lane_block(config, first_lane, lane_phase),
        )


def _place_input_wiring_extender_upper_row(
    schematic: mcschematic.MCSchematic,
    x: int,
    y: int,
    z: int,
    input_size: int,
    config: SchematicExportConfig,
    lane_phase: int,
) -> None:
    for pair_index, (base_x, has_second_target) in enumerate(
        _input_pair_bases(x, input_size)
    ):
        first_lane = pair_index * 2
        second_lane = first_lane + 1
        _place_input_targets(schematic, base_x, y, z, has_second_target)
        _place_repeater(
            schematic,
            (base_x, y, z + 1),
            config.input_repeater_south,
            config,
            support_block=config.slab_block,
        )
        _set_safe(
            schematic,
            (base_x + 1, y, z + 1),
            _lane_block(
                config, second_lane if pair_index == 0 else first_lane, lane_phase
            ),
        )
        _set_safe(
            schematic,
            (base_x + 2, y, z + 1),
            _wire(
                power=15,
                east="up",
                north="side",
                south="side",
                west="side" if pair_index == 0 else "none",
            ),
        )
        _set_safe(
            schematic,
            (base_x + 3, y, z + 1),
            _lane_block(config, second_lane, lane_phase),
        )
        if pair_index == 0:
            _set_safe(
                schematic,
                (base_x - 1, y, z + 2),
                _lane_block(config, second_lane, lane_phase),
            )
        _set_safe(
            schematic,
            (base_x, y, z + 2),
            _wire(power=15, east="up", north="side", south="side", west="none"),
        )
        _set_safe(
            schematic,
            (base_x + 1, y, z + 2),
            _lane_block(config, first_lane, lane_phase),
        )
        _place_repeater(
            schematic,
            (base_x + 2, y, z + 2),
            config.input_repeater_south,
            config,
            support_block=config.slab_block,
        )
        _set_safe(
            schematic,
            (base_x + 3, y, z + 2),
            _lane_block(config, second_lane, lane_phase),
        )
        _place_repeater(
            schematic,
            (base_x, y, z + 3),
            config.input_repeater_south,
            config,
            support_block=config.slab_block,
        )
        _set_safe(
            schematic,
            (base_x + 2, y, z + 3),
            _wire(power=14, east="none", north="side", south="none", west="side"),
        )
        _set_safe(
            schematic,
            (base_x, y, z + 4),
            _wire(power=14, east="none", north="side", south="none", west="side"),
        )


def _place_input_wiring(
    schematic: mcschematic.MCSchematic,
    layer_origin_x: int,
    layer_origin_y: int,
    layer_origin_z: int,
    input_size: int,
    output_size: int,
    config: SchematicExportConfig,
    lane_phase: int,
) -> None:
    if not config.emit_input_wiring:
        return

    wiring_z = layer_origin_z + 4
    if output_size > 0:
        _place_normal_input_wiring_row(
            schematic,
            layer_origin_x,
            layer_origin_y + 1,
            wiring_z,
            INPUT_RELAY_PERIOD,
            input_size,
            config,
            lane_phase=lane_phase,
            is_bottom_row=True,
        )
    for output_index in range(output_size):
        row_x = layer_origin_x + output_index + 1
        row_y = layer_origin_y + output_index + 2
        cycle = output_index % INPUT_RELAY_PERIOD
        if cycle == INPUT_RELAY_PERIOD - 1:
            _place_input_wiring_extender_lower_row(
                schematic, row_x, row_y, wiring_z, input_size, config, lane_phase
            )
        elif cycle == 0 and output_index > 0:
            _place_input_wiring_extender_upper_row(
                schematic, row_x, row_y, wiring_z, input_size, config, lane_phase
            )
        else:
            _place_normal_input_wiring_row(
                schematic,
                row_x,
                row_y,
                wiring_z,
                INPUT_RELAY_PERIOD - cycle,
                input_size,
                config,
                lane_phase=lane_phase,
                adds_extender_step=cycle == INPUT_RELAY_PERIOD - 2,
                is_top_row=output_index == output_size - 1,
                is_bottom_row=False,
            )


def _output_relay_dust_blocks() -> dict[str, str]:
    return {
        "full": _wire(east="side", north="side", south="side", west="side"),
        "north_up": _wire(north="side", west="up"),
        "north_south": _wire(north="side", south="side"),
        "south_up": _wire(south="side", west="up"),
        "east_north": _wire(east="side", north="side"),
        "east_north_south": _wire(east="side", north="side", south="side"),
    }


def _place_output_relay_down_a_tile(
    schematic: mcschematic.MCSchematic,
    origin: tuple[int, int, int],
    config: SchematicExportConfig,
    *,
    leading: bool = False,
    trailing: bool = False,
) -> None:
    """Place the black-lane half of a downward relay tile."""

    dust = _output_relay_dust_blocks()
    placements: list[BlockPlacement] = [
        (1, -2, -4, config.support_block),
        (1, -2, -3, config.support_block),
        (1, -2, -2, config.input_wiring_black_block),
        (0, -2, -1, _wire(east="side", west="up")),
        (0, -1, -4, config.support_block),
        (1, -1, -4, dust["south_up"]),
        (0, -1, -3, config.support_block),
        (1, -1, -3, config.output_repeater_north),
        (0, -1, -2, config.input_wiring_black_block),
        (0, 0, -4, _wire(east="side", south="side")),
        (0, 0, -3, config.output_repeater_south),
    ]
    if not trailing:
        placements.append((2, -2, -2, dust["full"]))
    if leading:
        placements.extend(
            (
                (1, -1, -2, dust["north_up"]),
                (1, 0, -2, config.support_block),
            )
        )
    else:
        placements.extend(
            (
                (0, 0, -2, dust["north_up"]),
                (1, -1, -2, dust["north_south"]),
                (1, 0, -2, config.support_block),
                (1, 0, -1, config.input_wiring_white_block),
            )
        )
    _place_pattern(schematic, origin, tuple(placements))


def _place_output_relay_down_b_tile(
    schematic: mcschematic.MCSchematic,
    origin: tuple[int, int, int],
    config: SchematicExportConfig,
) -> None:
    """Place the white-lane half of a downward relay tile."""

    dust = _output_relay_dust_blocks()
    placements: tuple[BlockPlacement, ...] = (
        (3, -2, -3, config.support_block),
        (3, -2, -2, config.slab_block),
        (3, -2, -1, config.input_wiring_white_block),
        (2, -1, -3, config.support_block),
        (3, -1, -3, dust["south_up"]),
        (2, -1, -2, config.slab_block),
        (3, -1, -2, config.output_repeater_north),
        (2, -1, -1, config.input_wiring_white_block),
        (3, -1, -1, dust["east_north"]),
        (2, 0, -3, _wire(east="side", south="side")),
        (2, 0, -2, config.output_repeater_south),
        (3, 0, -2, config.input_wiring_black_block),
        (3, 0, -1, config.support_block),
        (2, 0, -1, dust["north_up"]),
    )
    _place_pattern(schematic, origin, placements)


def _place_output_relay_down_boundary_tile(
    schematic: mcschematic.MCSchematic,
    origin: tuple[int, int, int],
    config: SchematicExportConfig,
) -> None:
    dust = _output_relay_dust_blocks()
    _place_pattern(
        schematic,
        origin,
        (
            (0, -1, -1, config.slab_block),
            (0, -1, 0, config.support_block),
            (0, 0, -2, dust["east_north_south"]),
            (0, 0, -1, config.output_repeater_south),
            (1, 0, -1, config.input_wiring_white_block),
        ),
    )


def _place_output_relay_tile(
    schematic: mcschematic.MCSchematic,
    origin: tuple[int, int, int],
    config: SchematicExportConfig,
    *,
    leading: bool,
    trailing: bool,
    include_white_lane: bool,
) -> None:
    """Place one repeated downward output relay tile.

    Each four-block tile contains one black-lane component and, when that lane
    exists at this relay height, the adjacent white-lane component. The caller
    clips the first and last tile so relay stations do not extend past active
    output lanes.
    """

    _place_output_relay_down_boundary_tile(schematic, origin, config)
    _place_output_relay_down_a_tile(
        schematic, origin, config, leading=leading, trailing=trailing
    )
    if include_white_lane:
        _place_output_relay_down_b_tile(schematic, origin, config)


def _copy_region(
    source: mcschematic.MCSchematic,
    target: mcschematic.MCSchematic,
    source_min: tuple[int, int, int],
    source_max: tuple[int, int, int],
    target_origin: tuple[int, int, int],
) -> tuple[int, int, int]:
    copied_max = target_origin
    for y in range(source_min[1], source_max[1] + 1):
        for z in range(source_min[2], source_max[2] + 1):
            for x in range(source_min[0], source_max[0] + 1):
                block = source.getBlockStateAt((x, y, z))
                if block == "minecraft:air":
                    continue
                target_position = (
                    target_origin[0] + x - source_min[0],
                    target_origin[1] + y - source_min[1],
                    target_origin[2] + z - source_min[2],
                )
                _set_safe(target, target_position, block)
                copied_max = (
                    max(copied_max[0], target_position[0]),
                    max(copied_max[1], target_position[1]),
                    max(copied_max[2], target_position[2]),
                )
    return copied_max


def _module_atlas_entries(
    config: SchematicExportConfig,
) -> tuple[
    tuple[str, mcschematic.MCSchematic, tuple[int, int, int], tuple[int, int, int]], ...
]:
    entries: list[
        tuple[str, mcschematic.MCSchematic, tuple[int, int, int], tuple[int, int, int]]
    ] = []

    def add(
        name: str,
        schematic: mcschematic.MCSchematic,
        source_min: tuple[int, int, int],
        source_max: tuple[int, int, int],
    ) -> None:
        entries.append((name, schematic, source_min, source_max))

    for name, output_index, lane_phase in (
        ("output_lane_black", 0, 0),
        ("output_lane_white", 0, 1),
        ("output_lane_black_relay_handoff", INPUT_RELAY_PERIOD - 1, 1),
        ("output_lane_white_relay_handoff", INPUT_RELAY_PERIOD - 1, 0),
    ):
        schematic = mcschematic.MCSchematic()
        source = (4, INPUT_RELAY_PERIOD if "relay_handoff" in name else 4, 0)
        _place_output_lane(
            schematic, source, output_index, config, floor_y=0, lane_phase=lane_phase
        )
        add(name, schematic, (0, 0, -4), (source[0] + source[1] + 2, source[1] + 1, 2))

    relay_modules: tuple[tuple[str, object], ...] = (
        (
            "relay_boundary",
            lambda schematic, origin: _place_output_relay_down_boundary_tile(
                schematic, origin, config
            ),
        ),
        (
            "relay_black_leading",
            lambda schematic, origin: _place_output_relay_down_a_tile(
                schematic, origin, config, leading=True
            ),
        ),
        (
            "relay_black_middle",
            lambda schematic, origin: _place_output_relay_down_a_tile(
                schematic, origin, config
            ),
        ),
        (
            "relay_black_trailing",
            lambda schematic, origin: _place_output_relay_down_a_tile(
                schematic, origin, config, trailing=True
            ),
        ),
        (
            "relay_white",
            lambda schematic, origin: _place_output_relay_down_b_tile(
                schematic, origin, config
            ),
        ),
        (
            "relay_tile_leading_pair",
            lambda schematic, origin: _place_output_relay_tile(
                schematic,
                origin,
                config,
                leading=True,
                trailing=False,
                include_white_lane=True,
            ),
        ),
        (
            "relay_tile_middle_pair",
            lambda schematic, origin: _place_output_relay_tile(
                schematic,
                origin,
                config,
                leading=False,
                trailing=False,
                include_white_lane=True,
            ),
        ),
        (
            "relay_tile_trailing_black_only",
            lambda schematic, origin: _place_output_relay_tile(
                schematic,
                origin,
                config,
                leading=False,
                trailing=True,
                include_white_lane=False,
            ),
        ),
    )
    relay_origin = (2, 3, 4)
    for name, place in relay_modules:
        schematic = mcschematic.MCSchematic()
        place(schematic, relay_origin)  # type: ignore[operator]
        add(name, schematic, (0, 0, -1), (8, 4, 5))

    return tuple(entries)


def build_module_atlas(
    config: SchematicExportConfig | None = None,
) -> mcschematic.MCSchematic:
    """Build a debug atlas with each output-wiring module isolated by an air gap."""

    config = config or SchematicExportConfig(emit_weights=False, emit_subtraction=False)
    atlas = mcschematic.MCSchematic()
    cursor_x = 1
    max_position = (0, 0, 0)
    for _name, schematic, source_min, source_max in _module_atlas_entries(config):
        module_max = _copy_region(
            schematic, atlas, source_min, source_max, (cursor_x, 1, 1)
        )
        max_position = (
            max(max_position[0], module_max[0]),
            max(max_position[1], module_max[1]),
            max(max_position[2], module_max[2]),
        )
        cursor_x = module_max[0] + MODULE_ATLAS_GAP + 1

    opposite_corner = (max_position[0] + 1, max_position[1] + 1, max_position[2] + 1)
    _set_safe(atlas, (0, 0, 0), MODULE_ATLAS_CORNER_BLOCK)
    _set_safe(atlas, opposite_corner, MODULE_ATLAS_CORNER_BLOCK)
    return atlas


def _place_output_lane(
    schematic: mcschematic.MCSchematic,
    source_block: tuple[int, int, int],
    output_index: int,
    config: SchematicExportConfig,
    *,
    floor_y: int,
    lane_phase: int,
) -> None:
    source_x, source_y, source_z = source_block
    subtractor_x = source_x - 1
    uses_black_lane = _is_black_lane(output_index, lane_phase)
    lane_z = source_z + (
        OUTPUT_BLACK_LANE_FORWARD_Z if uses_black_lane else OUTPUT_WHITE_LANE_FORWARD_Z
    )
    lane_block = _lane_block(config, output_index, lane_phase)
    relay_handoff = (
        source_y > floor_y and (source_y - floor_y) % INPUT_RELAY_PERIOD == 0
    )

    if config.emit_subtraction:
        _set_safe(schematic, source_block, config.support_block)
        _set_support(schematic, (subtractor_x, source_y - 1, source_z + 1), config)
        _set_safe(
            schematic,
            (subtractor_x, source_y, source_z + 1),
            config.comparator_south_off,
        )

    if not relay_handoff:
        repeater_z = source_z + OUTPUT_LANE_FORWARD_Z + (0 if uses_black_lane else 1)
        _place_repeater(
            schematic,
            (source_x, source_y, repeater_z),
            config.output_repeater_south,
            config,
            support_block=config.slab_block if uses_black_lane else lane_block,
        )
        if uses_black_lane:
            _set_safe(
                schematic,
                (source_x, source_y - 1, source_z + OUTPUT_LANE_FORWARD_Z + 1),
                lane_block,
            )
            _set_safe(
                schematic,
                (source_x, source_y, source_z + OUTPUT_LANE_FORWARD_Z + 1),
                _wire(north="side", south="side"),
            )
    _set_safe(schematic, (source_x, source_y, lane_z), _wire(east="side", south="side"))

    for y in range(source_y - 1, floor_y - 1, -1):
        step_down = source_y - y
        redstone_x = source_x + step_down
        _set_safe(schematic, (redstone_x - 1, y, lane_z), lane_block)
        _set_safe(schematic, (redstone_x, y, lane_z), _wire(east="side", west="up"))


def _output_block_x_offset(input_size: int, config: SchematicExportConfig) -> int:
    return input_size * config.input_pitch


def _output_source_block(
    layer_origin_x: int,
    layer_origin_y: int,
    layer_origin_z: int,
    input_size: int,
    config: SchematicExportConfig,
    output_index: int,
) -> tuple[int, int, int]:
    return (
        layer_origin_x + output_index + _output_block_x_offset(input_size, config),
        layer_origin_y + output_index + 1,
        layer_origin_z,
    )


def _output_relay_tile_x_range(
    layer_origin_x: int,
    layer_origin_y: int,
    input_size: int,
    output_size: int,
    config: SchematicExportConfig,
    *,
    relay_y: int,
    lane_phase: int,
) -> tuple[int, int]:
    output_x = layer_origin_x + _output_block_x_offset(input_size, config)
    relay_height = relay_y - layer_origin_y
    first_lane_index = relay_height - 1
    first_black_lane_phase = 0 if _is_black_lane(first_lane_index, lane_phase) else 2
    first_tile_x = output_x + relay_height - 1 + first_black_lane_phase
    last_lane_index = output_size - 1
    last_tile_x = output_x + 2 * last_lane_index + 1 - relay_height
    return first_tile_x, last_tile_x


def _place_output_wiring(
    schematic: mcschematic.MCSchematic,
    layer_origin_x: int,
    layer_origin_y: int,
    layer_origin_z: int,
    input_size: int,
    output_size: int,
    config: SchematicExportConfig,
    lane_phase: int,
) -> None:
    if not config.emit_output_wiring:
        return

    for output_index in range(output_size):
        _place_output_lane(
            schematic,
            _output_source_block(
                layer_origin_x,
                layer_origin_y,
                layer_origin_z,
                input_size,
                config,
                output_index,
            ),
            output_index,
            config,
            floor_y=layer_origin_y,
            lane_phase=lane_phase,
        )
    max_y = layer_origin_y + output_size
    for relay_y in range(
        layer_origin_y + INPUT_RELAY_PERIOD, max_y, INPUT_RELAY_PERIOD
    ):
        first_tile_x, max_active_x = _output_relay_tile_x_range(
            layer_origin_x,
            layer_origin_y,
            input_size,
            output_size,
            config,
            relay_y=relay_y,
            lane_phase=lane_phase,
        )
        tile_x = first_tile_x
        while tile_x <= max_active_x:
            _place_output_relay_tile(
                schematic,
                (tile_x, relay_y, layer_origin_z - 1),
                config,
                leading=tile_x == first_tile_x,
                trailing=tile_x + 4 > max_active_x,
                include_white_lane=tile_x + 1 <= max_active_x,
            )
            tile_x += 4


def _layer_io_bridge_template() -> tuple[BlockPlacement, ...]:
    return (
        (3, -4, 0, "minecraft:black_concrete"),
        (1, -4, 1, "minecraft:white_concrete"),
        (3, -4, 1, "minecraft:black_concrete"),
        (1, -4, 2, "minecraft:white_concrete"),
        (3, -4, 2, "minecraft:black_concrete"),
        (1, -4, 3, "minecraft:white_concrete"),
        (3, -4, 3, "minecraft:black_concrete"),
        (1, -4, 4, "minecraft:white_concrete"),
        (3, -4, 4, "minecraft:black_concrete"),
        (1, -4, 5, "minecraft:white_concrete"),
        (3, -4, 5, "minecraft:black_concrete"),
        (3, -4, 6, "minecraft:black_concrete"),
        (3, -3, -1, "minecraft:black_concrete"),
        (1, -3, 0, "minecraft:white_concrete"),
        (
            3,
            -3,
            0,
            "minecraft:repeater[delay=1,facing=north,locked=false,powered=false]",
        ),
        (
            1,
            -3,
            1,
            "minecraft:repeater[delay=1,facing=north,locked=false,powered=false]",
        ),
        (
            3,
            -3,
            1,
            _wire(north="side", south="side"),
        ),
        (
            1,
            -3,
            2,
            _wire(north="side", south="side"),
        ),
        (
            3,
            -3,
            2,
            _wire(north="side", south="side"),
        ),
        (
            1,
            -3,
            3,
            _wire(north="side", south="side"),
        ),
        (
            3,
            -3,
            3,
            _wire(north="side", south="side"),
        ),
        (
            1,
            -3,
            4,
            _wire(north="side", south="side"),
        ),
        (
            3,
            -3,
            4,
            _wire(north="side", south="side"),
        ),
        (
            1,
            -3,
            5,
            _wire(north="side", south="up"),
        ),
        (
            3,
            -3,
            5,
            _wire(north="side", south="side"),
        ),
        (1, -3, 6, "minecraft:white_concrete"),
        (
            3,
            -3,
            6,
            _wire(north="side", south="up"),
        ),
        (1, -3, 7, "minecraft:white_concrete"),
        (3, -3, 7, "minecraft:black_concrete"),
        (3, -3, 8, "minecraft:black_concrete"),
        (2, -2, -1, "minecraft:black_concrete"),
        (
            3,
            -2,
            -1,
            _wire(east="side", west="up"),
        ),
        (0, -2, 0, "minecraft:white_concrete"),
        (
            1,
            -2,
            0,
            _wire(east="side", west="up"),
        ),
        (4, -2, 0, "minecraft:white_concrete"),
        (0, -2, 1, "minecraft:light_gray_concrete"),
        (
            1,
            -2,
            6,
            _wire(north="side", south="side"),
        ),
        (
            1,
            -2,
            7,
            "minecraft:repeater[delay=1,facing=north,locked=false,powered=false]",
        ),
        (
            3,
            -2,
            7,
            _wire(north="side", south="side"),
        ),
        (1, -2, 8, "minecraft:white_concrete"),
        (
            3,
            -2,
            8,
            "minecraft:repeater[delay=1,facing=north,locked=false,powered=false]",
        ),
        (3, -2, 9, "minecraft:black_concrete"),
        (1, -1, -1, "minecraft:black_concrete"),
        (
            2,
            -1,
            -1,
            _wire(east="side", west="up"),
        ),
        (5, -1, -1, "minecraft:black_concrete"),
        (
            0,
            -1,
            0,
            _wire(east="side", south="side"),
        ),
        (1, -1, 0, "minecraft:smooth_stone_slab[type=top,waterlogged=false]"),
        (3, -1, 0, "minecraft:white_concrete"),
        (
            4,
            -1,
            0,
            _wire(east="side", west="up"),
        ),
        (
            0,
            -1,
            1,
            "minecraft:repeater[delay=1,facing=south,locked=false,powered=false]",
        ),
        (1, -1, 1, "minecraft:light_gray_concrete"),
        (0, -1, 2, "minecraft:light_gray_concrete"),
        (1, -1, 7, "minecraft:white_concrete"),
        (3, -1, 7, "minecraft:smooth_stone_slab[type=top,waterlogged=false]"),
        (
            1,
            -1,
            8,
            _wire(north="up", south="side"),
        ),
        (3, -1, 8, "minecraft:black_concrete"),
        (
            3,
            -1,
            9,
            _wire(north="up", south="side"),
        ),
        (
            1,
            0,
            -1,
            _wire(east="side", south="side"),
        ),
        (4, 0, -1, "minecraft:black_concrete"),
        (
            5,
            0,
            -1,
            _wire(east="side", west="up"),
        ),
        (
            1,
            0,
            0,
            "minecraft:repeater[delay=1,facing=south,locked=false,powered=false]",
        ),
        (2, 0, 0, "minecraft:white_concrete"),
        (
            3,
            0,
            0,
            _wire(east="side", west="up"),
        ),
        (
            1,
            0,
            1,
            "minecraft:repeater[delay=1,facing=south,locked=false,powered=false]",
        ),
        (2, 0, 1, "minecraft:light_gray_concrete"),
        (1, 0, 2, "minecraft:light_gray_concrete"),
        (1, 0, 6, "minecraft:target[power=0]"),
        (3, 0, 6, "minecraft:target[power=0]"),
        (5, 0, 6, "minecraft:target[power=0]"),
        (0, 0, 7, "minecraft:smooth_stone_slab[type=top,waterlogged=false]"),
        (
            1,
            0,
            7,
            _wire(east="up", north="side", south="side"),
        ),
        (2, 0, 7, "minecraft:white_concrete"),
        (
            3,
            0,
            7,
            "minecraft:repeater[delay=1,facing=south,locked=false,powered=false]",
        ),
        (4, 0, 7, "minecraft:smooth_stone_slab[type=top,waterlogged=false]"),
        (0, 0, 8, "minecraft:black_concrete"),
        (
            3,
            0,
            8,
            _wire(east="up", north="side", south="side"),
        ),
        (4, 0, 8, "minecraft:black_concrete"),
        (3, 1, -1, "minecraft:black_concrete"),
        (
            4,
            1,
            -1,
            _wire(east="side", west="up"),
        ),
        (
            2,
            1,
            0,
            _wire(east="side", south="side"),
        ),
        (3, 1, 0, "minecraft:smooth_stone_slab[type=top,waterlogged=false]"),
        (5, 1, 0, "minecraft:white_concrete"),
        (
            2,
            1,
            1,
            "minecraft:repeater[delay=1,facing=south,locked=false,powered=false]",
        ),
        (3, 1, 1, "minecraft:light_gray_concrete"),
        (2, 1, 2, "minecraft:light_gray_concrete"),
        (0, 1, 6, "minecraft:target[power=0]"),
        (2, 1, 6, "minecraft:target[power=0]"),
        (4, 1, 6, "minecraft:target[power=0]"),
        (
            0,
            1,
            7,
            "minecraft:repeater[delay=1,facing=south,locked=false,powered=false]",
        ),
        (1, 1, 7, "minecraft:smooth_stone_slab[type=top,waterlogged=false]"),
        (
            2,
            1,
            7,
            _wire(east="up", north="side", west="side"),
        ),
        (3, 1, 7, "minecraft:white_concrete"),
        (
            4,
            1,
            7,
            "minecraft:repeater[delay=1,facing=south,locked=false,powered=false]",
        ),
        (5, 1, 7, "minecraft:smooth_stone_slab[type=top,waterlogged=false]"),
        (
            0,
            1,
            8,
            _wire(east="up", north="side"),
        ),
        (1, 1, 8, "minecraft:black_concrete"),
        (
            4,
            1,
            8,
            _wire(east="up", north="side", west="side"),
        ),
        (5, 1, 8, "minecraft:black_concrete"),
    )


def _layer_io_bridge_blocks(
    config: SchematicExportConfig,
) -> tuple[BlockPlacement, ...]:
    return tuple(
        placement
        for placement in _layer_io_bridge_template()
        if placement[3]
        not in (
            "minecraft:target[power=0]",
            config.output_repeater_south,
            config.support_block,
        )
    )


_LEADING_BRIDGE_CARRY_OFFSETS = frozenset(
    {
        (2, 0, 0),
        (3, 0, 0),
        (2, 1, 0),
        (3, 1, -1),
        (3, 1, 0),
    }
)
_TRAILING_BRIDGE_CARRY_OFFSETS = _LEADING_BRIDGE_CARRY_OFFSETS | frozenset({(3, -1, 0)})


def _clips_layer_io_bridge_placement(
    placement: BlockPlacement,
    *,
    actual_lane: int,
    group_index: int,
    input_size: int,
) -> bool:
    """Clip connector carry blocks that would extend past the active lane edge."""

    dx, dy, dz, _block = placement
    relative_position = (dx, dy, dz)
    last_lane = input_size - 1
    penultimate_lane = input_size - 2

    if actual_lane >= input_size:
        return True

    leading_edge = (
        group_index == 0
        and actual_lane == 0
        and relative_position in _LEADING_BRIDGE_CARRY_OFFSETS
    )
    trailing_edge = (
        actual_lane == last_lane and relative_position in _TRAILING_BRIDGE_CARRY_OFFSETS
    )
    penultimate_edge = actual_lane == penultimate_lane and relative_position in {
        (5, 1, 0),
        (1, -1, 0),
    }
    side_carry_overrun = actual_lane >= input_size - 3 and relative_position == (
        3,
        1,
        0,
    )
    return leading_edge or trailing_edge or penultimate_edge or side_carry_overrun


def _place_layer_io_bridge(
    schematic: mcschematic.MCSchematic,
    next_layer_origin_x: int,
    next_layer_origin_y: int,
    next_layer_origin_z: int,
    next_input_size: int,
    config: SchematicExportConfig,
    lane_phase: int,
) -> None:
    template = _layer_io_bridge_blocks(config)
    lanes_per_bridge_step = 2
    x_stride = lanes_per_bridge_step * config.input_pitch
    origin_y = next_layer_origin_y + 2
    origin_z = next_layer_origin_z - 2
    phase_template_offset = 0 if lane_phase == 1 else 1
    group_count = math.ceil(
        (next_input_size + phase_template_offset) / lanes_per_bridge_step
    )
    for group_index in range(group_count):
        group_input_start = group_index * lanes_per_bridge_step
        group_origin_x = (
            next_layer_origin_x
            + group_index * x_stride
            - phase_template_offset * config.input_pitch
        )
        group_origin = (group_origin_x, origin_y, origin_z)
        for placement in template:
            dx, _dy, _dz, _block = placement
            actual_lane = (
                group_input_start + dx // config.input_pitch - phase_template_offset
            )
            if actual_lane < 0:
                continue
            if _clips_layer_io_bridge_placement(
                placement,
                actual_lane=actual_lane,
                group_index=group_index,
                input_size=next_input_size,
            ):
                continue
            _place_if_air(schematic, group_origin, placement)


def build_schematic(
    model: DiscreteMCNN, config: SchematicExportConfig | None = None
) -> mcschematic.MCSchematic:
    """Build a dense diagonally staggered debug schematic from a discrete model."""

    config = config or SchematicExportConfig()
    schematic = mcschematic.MCSchematic()
    layer_placements = _layer_placements(model, config)

    for layer_index, (
        layer,
        (input_size, layer_origin_x, layer_origin_y, input_lane_phase),
    ) in enumerate(zip(model.layers, layer_placements)):
        layer_origin_z = 0
        output_size = int(layer.weight.shape[0])
        output_lane_phase = (input_lane_phase + input_size) % 2

        _place_input_wiring(
            schematic,
            layer_origin_x,
            layer_origin_y,
            layer_origin_z,
            input_size,
            output_size,
            config,
            input_lane_phase,
        )

        for output_index in range(output_size):
            connections: list[tuple[int, str, int]] = []
            inactive_total = 0
            for input_index in range(input_size):
                weight = int(layer.weight[output_index, input_index].item())
                subtractions = comparator_subtractions(weight)
                encoding = subtractions.encoding
                if (
                    encoding.mode != "C"
                    and furnace_signal_items(encoding.value) is None
                ):
                    raise ValueError(
                        f"layer {layer_index} output {output_index} input {input_index} "
                        f"requires unrepresentable furnace signal {encoding.value}"
                    )
                inactive_total += subtractions.inactive
                connections.append((input_index, encoding.mode, encoding.value))

            effective_bias = int(layer.bias[output_index].item())
            physical_bias = effective_bias + inactive_total
            if physical_bias > MAX_BIAS_SIGNAL_STRENGTH:
                raise ValueError(
                    f"layer {layer_index} output {output_index} requires physical bias "
                    f"{physical_bias}, exceeding maximum {MAX_BIAS_SIGNAL_STRENGTH}"
                )
            if not is_representable_bias_signal(physical_bias):
                raise ValueError(
                    f"layer {layer_index} output {output_index} requires physical bias "
                    f"{physical_bias}, which is not representable by the capped barrel encoding"
                )

            _place_neuron(
                schematic,
                layer_origin_x,
                layer_origin_y,
                layer_origin_z,
                output_index,
                physical_bias,
                connections,
                config,
            )

        _place_output_wiring(
            schematic,
            layer_origin_x,
            layer_origin_y,
            layer_origin_z,
            input_size,
            output_size,
            config,
            output_lane_phase,
        )

    if config.emit_layer_io_connectors:
        if not config.emit_input_wiring or not config.emit_output_wiring:
            raise ValueError(
                "layer I/O connectors require both input and output wiring"
            )
        for (
            next_input_size,
            next_origin_x,
            next_origin_y,
            next_lane_phase,
        ) in layer_placements[1:]:
            _place_layer_io_bridge(
                schematic,
                next_origin_x,
                next_origin_y,
                0,
                next_input_size,
                config,
                next_lane_phase,
            )

    return schematic


def save_schematic(
    model: DiscreteMCNN,
    output_path: str | Path,
    *,
    config: SchematicExportConfig | None = None,
    version: mcschematic.Version = mcschematic.Version.JE_1_21_5,
) -> Path:
    """Build and save a ``.schem`` file for a dense diagonally staggered layout."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schematic = build_schematic(model, config)
    schematic.save(str(output_path.parent), output_path.stem, version)
    return output_path.with_suffix(".schem")


def save_module_atlas(
    output_path: str | Path,
    *,
    config: SchematicExportConfig | None = None,
    version: mcschematic.Version = mcschematic.Version.JE_1_21_5,
) -> Path:
    """Build and save an isolated module atlas for editing wiring templates."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schematic = build_module_atlas(config)
    schematic.save(str(output_path.parent), output_path.stem, version)
    return output_path.with_suffix(".schem")
