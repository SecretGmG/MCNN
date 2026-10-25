"""Minimal KataGo analysis client and project-wide 7x7 configuration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .go_rules import BLACK, BOARD_SIZE, GoPosition, move_to_gtp


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KATAGO = ROOT / "tools/katago-build/katago"
DEFAULT_ENGINE_CONFIG = ROOT / "tools/katago/go7x7.cfg"
DEFAULT_NETWORK = ROOT / "tools/katago/models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
KATAGO_RULES = {
    "hasButton": False,
    "ko": "POSITIONAL",
    "scoring": "TERRITORY",
    "suicide": False,
    "tax": "NONE",
    "whiteHandicapBonus": "0",
    "friendlyPassOk": False,
}


class KataGoAnalysis:
    """Line-oriented wrapper around KataGo's JSON analysis process."""

    def __init__(self, executable: Path, config: Path, network: Path) -> None:
        self.process = subprocess.Popen(
            [str(executable), "analysis", "-config", str(config), "-model", str(network)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def query(self, payload: dict[str, object]) -> dict[str, object]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("KataGo process has no standard streams")
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError(f"KataGo exited unexpectedly with code {self.process.poll()}")
        decoded = json.loads(response)
        if "error" in decoded:
            raise RuntimeError(f"KataGo analysis error: {decoded['error']}")
        return decoded

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        self.process.wait(timeout=10)


def query_payload(
    moves: list[list[str]],
    visits: int,
    query_id: int,
    *,
    include_policy: bool = True,
    include_ownership: bool = False,
    include_ownership_stdev: bool = False,
    komi: float = 0.0,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": str(query_id),
        "boardXSize": BOARD_SIZE,
        "boardYSize": BOARD_SIZE,
        "rules": KATAGO_RULES,
        "komi": komi,
        "moves": moves,
        "includePolicy": include_policy,
        "maxVisits": visits,
    }
    if include_ownership:
        payload["includeOwnership"] = True
    if include_ownership_stdev:
        payload["includeOwnershipStdev"] = True
    return payload


def history_move(position: GoPosition, move: int) -> list[str]:
    return ["B" if position.to_play == BLACK else "W", move_to_gtp(move)]


def future_score_value(response: dict[str, object], position: GoPosition) -> float:
    """Return side-to-move score lead excluding historical prisoners."""
    score_lead = float(response["rootInfo"]["scoreLead"])
    prisoner_advantage = (
        position.black_prisoners - position.white_prisoners
    ) * position.to_play
    return score_lead - prisoner_advantage
