"""Evaluate a physical model on a fixed noisy-KataGo score benchmark."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts.common.plotting import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np

from mcnn.go_inference import load_successor_model, select_successor_move
from mcnn.go_match import relative_final_score, replay_history, sample_policy_move
from mcnn.go_rules import BLACK
from mcnn.katago import (
    DEFAULT_ENGINE_CONFIG,
    DEFAULT_KATAGO,
    DEFAULT_NETWORK,
    KataGoAnalysis,
    history_move,
    query_payload,
)


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    margins = np.asarray([record["final_score_margin"] for record in records], dtype=np.float64)
    initial = np.asarray([record["initial_katago_score_margin"] for record in records])
    winning = initial > 0
    return {
        "games": len(records),
        "mean_final_score_margin": float(margins.mean()),
        "median_final_score_margin": float(np.median(margins)),
        "p10_final_score_margin": float(np.quantile(margins, 0.1)),
        "wins": int((margins > 0).sum()),
        "win_rate": float((margins > 0).mean()),
        "mean_change_from_initial": float((margins - initial).mean()),
        "winning_starts": int(winning.sum()),
        "winning_conversions": int(((margins > 0) & winning).sum()),
        "winning_conversion_rate": float(
            ((margins > 0) & winning).sum() / max(1, winning.sum())
        ),
    }


def evaluate(
    model_path: Path,
    suite_path: Path,
    visits: int,
    temperature: float,
    noise: float,
    dirichlet_alpha: float,
    seed: int,
    katago: Path,
    engine_config: Path,
    network: Path,
) -> dict[str, object]:
    suite = json.loads(suite_path.read_text())
    model = load_successor_model(str(model_path)).eval()
    engine = KataGoAnalysis(katago, engine_config, network)
    records: list[dict[str, object]] = []
    query_id = 0
    try:
        for game, scenario in enumerate(suite["scenarios"]):
            model_color = BLACK if scenario["model_color"] == "black" else -BLACK
            position = replay_history(scenario["moves"])
            moves = [move.copy() for move in scenario["moves"]]
            continuation: list[dict[str, object]] = []
            rng = np.random.default_rng(seed + game)
            for _ in range(120 - len(moves)):
                if position.to_play == model_color:
                    move = select_successor_move(model, position)
                    actor = "model"
                else:
                    response = engine.query(query_payload(moves, visits, query_id, komi=7.5))
                    query_id += 1
                    move = sample_policy_move(
                        response,
                        position,
                        rng,
                        temperature=temperature,
                        noise=noise,
                        dirichlet_alpha=dirichlet_alpha,
                    )
                    actor = "katago"
                recorded = history_move(position, move)
                successor = position.play(move)
                if successor is None:
                    raise RuntimeError(f"Illegal {actor} move in {scenario['id']}: {recorded}")
                moves.append(recorded)
                continuation.append({"actor": actor, "color": recorded[0], "move": recorded[1]})
                position = successor
                if position.is_finished:
                    break
            records.append(
                {
                    "scenario": scenario["id"],
                    "category": scenario["category"],
                    "model_color": scenario["model_color"],
                    "initial_ply": scenario["initial_ply"],
                    "initial_katago_score_margin": scenario["initial_katago_score_margin"],
                    "final_score_margin": relative_final_score(position, model_color),
                    "finished": position.is_finished,
                    "final_ply": len(moves),
                    "continuation": continuation,
                    "moves": moves,
                }
            )
    finally:
        engine.close()

    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_category[str(record["category"])].append(record)
    overall = _aggregate(records)
    return {
        "objective": {
            "name": "mean_final_score_margin",
            "direction": "maximize",
            "value": overall["mean_final_score_margin"],
        },
        "model": str(model_path),
        "suite": str(suite_path),
        "opponent": {
            "visits": visits,
            "temperature": temperature,
            "dirichlet_noise": noise,
            "dirichlet_alpha": dirichlet_alpha,
            "seed": seed,
        },
        "overall": overall,
        "by_category": {category: _aggregate(group) for category, group in by_category.items()},
        "records": records,
    }


def plot(result: dict[str, object], output: Path) -> None:
    records = sorted(result["records"], key=lambda row: (row["category"], row["scenario"]))
    colors = {"balanced": "#3274a1", "defensive": "#c44e52", "winning": "#55a868"}
    y = np.arange(len(records))
    initial = [record["initial_katago_score_margin"] for record in records]
    final = [record["final_score_margin"] for record in records]
    figure, (match_axis, summary_axis) = plt.subplots(
        1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.7, 1]}
    )
    for index, record in enumerate(records):
        color = colors[record["category"]]
        match_axis.plot([initial[index], final[index]], [index, index], color=color, alpha=0.45)
        match_axis.scatter(initial[index], index, facecolors="none", edgecolors=color, s=42)
        match_axis.scatter(final[index], index, color=color, s=42)
    match_axis.axvline(0, color="#222222", linewidth=1)
    match_axis.set_yticks(y, [record["scenario"] for record in records])
    match_axis.set_xlabel("Model-relative score margin")
    match_axis.set_title("Start (open) to final (filled)")
    categories = [
        category
        for category in ("balanced", "defensive", "winning")
        if category in result["by_category"]
    ]
    means = [result["by_category"][category]["mean_final_score_margin"] for category in categories]
    summary_axis.bar(categories, means, color=[colors[category] for category in categories])
    summary_axis.axhline(0, color="#222222", linewidth=1)
    summary_axis.set_ylabel("Mean final score margin")
    summary_axis.set_title(f"Overall: {result['objective']['value']:.1f}")
    for axis in (match_axis, summary_axis):
        axis.grid(axis="x" if axis is match_axis else "y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visits", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=55001)
    parser.add_argument("--katago", type=Path, default=DEFAULT_KATAGO)
    parser.add_argument("--engine-config", type=Path, default=DEFAULT_ENGINE_CONFIG)
    parser.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    args = parser.parse_args()
    if args.visits < 1:
        raise ValueError("Visits must be positive")
    result = evaluate(
        args.model,
        args.suite,
        args.visits,
        args.temperature,
        args.noise,
        args.dirichlet_alpha,
        args.seed,
        args.katago,
        args.engine_config,
        args.network,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    plot(result, args.output.with_suffix(".png"))
    print(
        json.dumps(
            {
                "objective": result["objective"],
                "overall": result["overall"],
                "by_category": result["by_category"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
