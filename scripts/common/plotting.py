"""Consistent headless Matplotlib setup for report commands."""

from __future__ import annotations

import os


def configure_matplotlib() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mcnn-matplotlib")
