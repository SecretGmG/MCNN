"""Minecraft comparator encoding for effective connection weights.

Each nonzero weight is encoded as one of three comparator modes that map to
physical redstone subtraction modules:

- Mode A: inactive subtracts 15, active subtracts a nonzero ``value``.
  Effective weight is ``15 - value``.
- Mode B: inactive subtracts a nonzero ``value``, active subtracts 15.
  Effective weight is ``value - 15``.
- Mode C: zero effective weight. No comparator module is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComparatorEncoding:
    """Minecraft comparator encoding for one effective connection weight."""

    mode: str
    value: int


@dataclass(frozen=True)
class ComparatorSubtractions:
    """Subtractions made by one physical connection module.

    ``inactive`` is the amount subtracted when the binary input is 0.
    ``active`` is the amount subtracted when the binary input is 1.
    The effective model weight is therefore ``inactive - active``.
    """

    encoding: ComparatorEncoding
    inactive: int
    active: int


def encode_comparator_weight(weight: int) -> ComparatorEncoding:
    """Encode an effective integer weight as a comparator mode and value.

    Mode A subtracts 15 by default and ``value`` when active, giving
    ``weight = 15 - value``. Mode B subtracts ``value`` by default and 15 when
    active, giving ``weight = value - 15``. ``value`` must be nonzero because a
    zero-signal furnace behaves like a normal block in the redstone circuit.
    Mode C is the zero-weight no-op: no comparator connection module is emitted,
    reducing the required bias signal.
    """

    if weight > 0:
        return ComparatorEncoding(mode="A", value=15 - weight)
    if weight < 0:
        return ComparatorEncoding(mode="B", value=weight + 15)
    return ComparatorEncoding(mode="C", value=0)


def comparator_subtractions(weight: int) -> ComparatorSubtractions:
    """Return the exact Minecraft subtraction behavior for a model weight."""

    encoding = encode_comparator_weight(weight)
    if encoding.mode == "A":
        return ComparatorSubtractions(
            encoding=encoding, inactive=15, active=encoding.value
        )
    if encoding.mode == "B":
        return ComparatorSubtractions(
            encoding=encoding, inactive=encoding.value, active=15
        )
    if encoding.mode == "C":
        return ComparatorSubtractions(encoding=encoding, inactive=0, active=0)
    raise ValueError(f"unknown comparator mode {encoding.mode!r}")


def inactive_subtraction(weight: int) -> int:
    """Return the inactive subtraction for a given effective weight."""
    return comparator_subtractions(weight).inactive
