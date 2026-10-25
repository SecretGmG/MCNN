"""Generate fixed noisy-KataGo starts for score-margin model evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mcnn.go_match import relative_katago_score, replay_history, sample_policy_move
from mcnn.go_rules import BLACK, GoPosition
from mcnn.katago import (
    DEFAULT_ENGINE_CONFIG,
    DEFAULT_KATAGO,
    DEFAULT_NETWORK,
    KATAGO_RULES,
    KataGoAnalysis,
    history_move,
    query_payload,
)


@dataclass(frozen=True)
class MatchSuiteConfig:
    scenarios_per_category: int = 4
    source_games: int = 8
    trajectory_visits: int = 6
    classification_visits: int = 48
    max_opening_plies: int = 52
    snapshot_stride: int = 4
    opening_temperature: float = 1.5
    opening_noise: float = 0.35
    dirichlet_alpha: float = 0.3
    balanced_lead: float = 8.0
    decisive_lead_min: float = 5.0
    decisive_lead_max: float = 18.0
    seed: int = 44001


def _candidate_category(
    lead: float,
    ply: int,
    counts: dict[str, int],
    cfg: MatchSuiteConfig,
) -> tuple[str, int] | None:
    limit = cfg.scenarios_per_category
    if abs(lead) <= cfg.balanced_lead and counts["balanced"] < limit:
        color = BLACK if counts["balanced"] % 2 == 0 else -BLACK
        return "balanced", color
    if not cfg.decisive_lead_min <= abs(lead) <= cfg.decisive_lead_max:
        return None
    ahead = BLACK if lead > 0 else -BLACK
    behind = -ahead
    # Later winning starts directly test lead conversion and passing.
    if counts["winning"] < limit and (ply >= 28 or counts["defensive"] >= limit):
        return "winning", ahead
    if counts["defensive"] < limit:
        return "defensive", behind
    if counts["winning"] < limit:
        return "winning", ahead
    return None


def generate_suite(
    cfg: MatchSuiteConfig,
    katago: Path,
    engine_config: Path,
    network: Path,
) -> dict[str, object]:
    engine = KataGoAnalysis(katago, engine_config, network)
    rng = np.random.default_rng(cfg.seed)
    candidates: list[dict[str, object]] = []
    query_id = 0
    try:
        for source_game in range(cfg.source_games):
            position = GoPosition()
            moves: list[list[str]] = []
            game_rng = np.random.default_rng(cfg.seed + source_game + 1)
            for ply in range(cfg.max_opening_plies + 1):
                response = engine.query(
                    query_payload(moves, cfg.trajectory_visits, query_id, komi=7.5)
                )
                query_id += 1
                if ply >= 12 and ply % cfg.snapshot_stride == source_game % cfg.snapshot_stride:
                    candidates.append(
                        {
                            "source_game": source_game,
                            "ply": ply,
                            "moves": [move.copy() for move in moves],
                            "rough_black_lead": relative_katago_score(response, position, BLACK),
                        }
                    )
                if ply == cfg.max_opening_plies:
                    break
                move = sample_policy_move(
                    response,
                    position,
                    game_rng,
                    temperature=cfg.opening_temperature,
                    noise=cfg.opening_noise,
                    dirichlet_alpha=cfg.dirichlet_alpha,
                    suppress_pass=ply < 40,
                )
                successor = position.play(move)
                if successor is None:
                    raise RuntimeError(f"KataGo opening produced illegal move {move}")
                moves.append(history_move(position, move))
                position = successor
                if position.is_finished:
                    break

        rng.shuffle(candidates)
        candidates.sort(
            key=lambda row: (
                abs(float(row["rough_black_lead"])) > cfg.decisive_lead_max + 8,
                int(row["ply"]) < 20,
            )
        )
        counts = {"balanced": 0, "defensive": 0, "winning": 0}
        scenarios: list[dict[str, object]] = []
        source_uses: dict[int, int] = {}
        for candidate in candidates:
            if all(count == cfg.scenarios_per_category for count in counts.values()):
                break
            source_game = int(candidate["source_game"])
            if source_uses.get(source_game, 0) >= 3:
                continue
            moves = candidate["moves"]
            position = replay_history(moves)
            response = engine.query(
                query_payload(moves, cfg.classification_visits, query_id, komi=7.5)
            )
            query_id += 1
            black_lead = relative_katago_score(response, position, BLACK)
            choice = _candidate_category(black_lead, int(candidate["ply"]), counts, cfg)
            if choice is None:
                continue
            category, model_color = choice
            model_lead = black_lead if model_color == BLACK else -black_lead
            scenario_id = f"{category}_{counts[category] + 1:02d}"
            scenarios.append(
                {
                    "id": scenario_id,
                    "category": category,
                    "model_color": "black" if model_color == BLACK else "white",
                    "initial_ply": int(candidate["ply"]),
                    "initial_katago_score_margin": round(float(model_lead), 3),
                    "moves": moves,
                }
            )
            counts[category] += 1
            source_uses[source_game] = source_uses.get(source_game, 0) + 1
    finally:
        engine.close()

    if any(count != cfg.scenarios_per_category for count in counts.values()):
        raise RuntimeError(f"Could not fill match suite: {counts}")
    scenarios.sort(key=lambda row: (row["category"], row["id"]))
    return {
        "version": 1,
        "rules": KATAGO_RULES,
        "komi": 7.5,
        "generation": asdict(cfg),
        "objective": "maximize mean final score margin from the model's perspective",
        "scenario_counts": counts,
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios-per-category", type=int, default=4)
    parser.add_argument("--source-games", type=int, default=MatchSuiteConfig.source_games)
    parser.add_argument("--trajectory-visits", type=int, default=MatchSuiteConfig.trajectory_visits)
    parser.add_argument("--classification-visits", type=int, default=48)
    parser.add_argument("--seed", type=int, default=44001)
    parser.add_argument("--katago", type=Path, default=DEFAULT_KATAGO)
    parser.add_argument("--engine-config", type=Path, default=DEFAULT_ENGINE_CONFIG)
    parser.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    args = parser.parse_args()
    cfg = MatchSuiteConfig(
        scenarios_per_category=args.scenarios_per_category,
        source_games=args.source_games,
        trajectory_visits=args.trajectory_visits,
        classification_visits=args.classification_visits,
        seed=args.seed,
    )
    if min(
        cfg.scenarios_per_category,
        cfg.source_games,
        cfg.trajectory_visits,
        cfg.classification_visits,
    ) < 1:
        raise ValueError("Scenario, game, and visit counts must be positive")
    suite = generate_suite(cfg, args.katago, args.engine_config, args.network)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2) + "\n")
    print(json.dumps({key: suite[key] for key in ("objective", "scenario_counts")}, indent=2))


if __name__ == "__main__":
    main()
