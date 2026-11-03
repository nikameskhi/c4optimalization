"""Run repeated strategy evaluations and save CSV, JSON, and SVG reports."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from connect_four import EvaluationResult, evaluate_strategy
from connect_four_core import Strategy
from heuristic_strategy import HeuristicWeights, choose_move
from mcts_strategy import MCTSStrategy
from minimax_strategy import MinimaxStrategy
from monte_carlo_strategy import MonteCarloStrategy

STRATEGIES = ("heuristic", "minimax", "monte-carlo", "mcts")


@dataclass(frozen=True)
class ExperimentRun:
    strategy: str
    run: int
    seed: int
    parameters: str
    wins: int
    draws: int
    losses: int
    games: int
    score: float
    first_player_games: int
    second_player_games: int
    started_first_games: int
    started_second_games: int
    elapsed_seconds: float


def _strategy_factory(
    name: str,
    weights: HeuristicWeights,
    depth: int,
    simulations: int,
    iterations: int,
    exploration: float,
) -> tuple[Callable[[int], Strategy], dict[str, object]]:
    if name == "heuristic":
        return (
            lambda _seed: lambda board, player: choose_move(board, player, weights),
            {"weights": asdict(weights)},
        )
    if name == "minimax":
        return (
            lambda _seed: MinimaxStrategy(depth=depth, weights=weights),
            {"depth": depth, "weights": asdict(weights)},
        )
    if name == "monte-carlo":
        return (
            lambda seed: MonteCarloStrategy(simulations=simulations, seed=seed),
            {"simulations_per_move": simulations},
        )
    if name == "mcts":
        return (
            lambda seed: MCTSStrategy(
                iterations=iterations, exploration=exploration, seed=seed
            ),
            {"iterations_per_move": iterations, "exploration": exploration},
        )
    raise ValueError(f"unknown strategy: {name}")


def run_experiments(
    strategies: Sequence[str],
    runs: int,
    games_per_side: int,
    seed: int,
    output_dir: Path,
    depth: int = 4,
    simulations: int = 100,
    iterations: int = 500,
    exploration: float = math.sqrt(2),
    center_weight: float = 0.5,
    open_windows_weight: float = 0.5,
) -> tuple[list[ExperimentRun], dict[str, dict[str, float]]]:
    if not strategies:
        raise ValueError("select at least one strategy")
    if any(strategy not in STRATEGIES for strategy in strategies):
        raise ValueError(f"strategies must be selected from: {', '.join(STRATEGIES)}")
    if type(runs) is not int or runs <= 0:
        raise ValueError("runs must be a positive integer")
    if type(games_per_side) is not int or games_per_side <= 0:
        raise ValueError("games_per_side must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    weights = HeuristicWeights(center_weight, open_windows_weight)
    rows = []
    for strategy_name in strategies:
        factory, parameters = _strategy_factory(
            strategy_name, weights, depth, simulations, iterations, exploration
        )
        for run_number in range(runs):
            run_seed = seed + run_number * 10_000
            start = time.perf_counter()
            # Reuse each run's seed range across strategies for paired comparisons.
            result: EvaluationResult = evaluate_strategy(
                factory, games_per_side=games_per_side, seed=run_seed
            )
            elapsed = time.perf_counter() - start
            rows.append(
                ExperimentRun(
                    strategy=strategy_name,
                    run=run_number + 1,
                    seed=run_seed,
                    parameters=json.dumps(parameters, sort_keys=True),
                    wins=result.wins,
                    draws=result.draws,
                    losses=result.losses,
                    games=result.games,
                    score=result.score,
                    first_player_games=result.first_player_games,
                    second_player_games=result.second_player_games,
                    started_first_games=result.started_first_games,
                    started_second_games=result.started_second_games,
                    elapsed_seconds=elapsed,
                )
            )

    summaries = summarize_runs(rows)
    save_results(rows, summaries, output_dir, runs, games_per_side, seed)
    return rows, summaries


def summarize_runs(rows: Sequence[ExperimentRun]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[ExperimentRun]] = {}
    for row in rows:
        grouped.setdefault(row.strategy, []).append(row)

    summaries = {}
    for strategy, strategy_runs in grouped.items():
        scores = [row.score for row in strategy_runs]
        # Population deviation describes the observed set of independent runs.
        summaries[strategy] = {
            "runs": float(len(strategy_runs)),
            "mean_score": statistics.mean(scores),
            "score_standard_deviation": statistics.pstdev(scores),
            "mean_wins": statistics.mean(row.wins for row in strategy_runs),
            "mean_draws": statistics.mean(row.draws for row in strategy_runs),
            "mean_losses": statistics.mean(row.losses for row in strategy_runs),
            "mean_elapsed_seconds": statistics.mean(
                row.elapsed_seconds for row in strategy_runs
            ),
            "elapsed_standard_deviation": statistics.pstdev(
                row.elapsed_seconds for row in strategy_runs
            ),
        }
    return summaries


def _write_svg(
    summaries: dict[str, dict[str, float]],
    path: Path,
    metric: str = "mean_score",
    deviation_metric: str = "score_standard_deviation",
    title: str = "Connect Four strategy comparison",
    axis_label: str = "Mean score (win=1, draw=0.5, loss=0)",
) -> None:
    # Keep chart output dependency-free so reports work in a standard Python install.
    width, height = 900, 520
    left, right, top, bottom = 90, 35, 75, 110
    plot_width = width - left - right
    plot_height = height - top - bottom
    colors = ("#4472C4", "#70AD47", "#ED7D31", "#8064A2")
    items = list(summaries.items())
    band_width = plot_width / max(len(items), 1)
    bar_width = min(90, band_width * 0.55)
    maximum = 1.0 if metric == "mean_score" else max(
        (summary[metric] + summary[deviation_metric] for _, summary in items),
        default=1.0,
    ) * 1.1
    maximum = max(maximum, 1e-9)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" '
        'font-family="sans-serif" font-size="22" font-weight="bold">'
        f"{html.escape(title)}</text>",
        f'<desc>{html.escape(axis_label)} averaged over repeated runs with '
        "one-standard-deviation error bars.</desc>",
    ]

    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_height * (1 - value / maximum)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" '
            f'y2="{y:.1f}" stroke="#d9d9d9"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="13">{value:.2f}</text>'
        )
    parts.extend((
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        'stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_height}" '
        f'x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333"/>',
        f'<text x="22" y="{top + plot_height / 2}" transform="rotate(-90 22 '
        f'{top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" '
        f'font-size="14">{html.escape(axis_label)}</text>',
    ))

    for index, (strategy, summary) in enumerate(items):
        center_x = left + band_width * (index + 0.5)
        value = summary[metric]
        deviation = summary[deviation_metric]
        bar_height = value / maximum * plot_height
        bar_x = center_x - bar_width / 2
        bar_y = top + plot_height - bar_height
        color = colors[index % len(colors)]
        lower = max(0.0, value - deviation)
        upper = min(maximum, value + deviation)
        error_top = top + plot_height * (1 - upper / maximum)
        error_bottom = top + plot_height * (1 - lower / maximum)
        label = html.escape(strategy)
        value_label = (
            f"{value:.3f} ± {deviation:.3f}"
            if metric == "mean_score"
            else f"{value:.2f} ± {deviation:.2f} s"
        )
        parts.extend((
            f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="{color}"/>',
            f'<line x1="{center_x:.1f}" y1="{error_top:.1f}" '
            f'x2="{center_x:.1f}" y2="{error_bottom:.1f}" stroke="#222" '
            'stroke-width="2"/>',
            f'<line x1="{center_x - 8:.1f}" y1="{error_top:.1f}" '
            f'x2="{center_x + 8:.1f}" y2="{error_top:.1f}" stroke="#222" '
            'stroke-width="2"/>',
            f'<line x1="{center_x - 8:.1f}" y1="{error_bottom:.1f}" '
            f'x2="{center_x + 8:.1f}" y2="{error_bottom:.1f}" stroke="#222" '
            'stroke-width="2"/>',
            f'<text x="{center_x:.1f}" y="{top + plot_height + 25}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="14">{label}</text>',
            f'<text x="{center_x:.1f}" y="{max(top + 16, bar_y - 10):.1f}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="13">'
            f'{value_label}</text>',
        ))

    parts.extend((
        f'<text x="{width / 2}" y="{height - 24}" text-anchor="middle" '
        'font-family="sans-serif" font-size="13">'
        f'{int(next(iter(summaries.values()))["runs"] if summaries else 0)} runs per strategy; '
        "error bars show one standard deviation</text>",
        "</svg>",
    ))
    path.write_text("\n".join(parts), encoding="utf-8")


def save_results(
    rows: Sequence[ExperimentRun],
    summaries: dict[str, dict[str, float]],
    output_dir: Path,
    runs: int,
    games_per_side: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save detail, aggregate tables, and charts so results are easy to inspect.
    csv_path = output_dir / "strategy_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    summary_csv_path = output_dir / "strategy_summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["strategy", *next(iter(summaries.values())).keys()]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for strategy, summary in summaries.items():
            writer.writerow({"strategy": strategy, **summary})

    report = {
        "baseline": "random_legal",
        "runs_per_strategy": runs,
        "games_per_side_per_run": games_per_side,
        "seed": seed,
        "score_definition": "win=1, draw=0.5, loss=0",
        "runs": [asdict(row) for row in rows],
        "summary": summaries,
    }
    (output_dir / "strategy_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_svg(summaries, output_dir / "strategy_scores.svg")
    _write_svg(
        summaries,
        output_dir / "strategy_runtime.svg",
        metric="mean_elapsed_seconds",
        deviation_metric="elapsed_standard_deviation",
        title="Connect Four evaluation time",
        axis_label="Mean elapsed time per run (seconds)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=list(STRATEGIES),
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--games-per-side", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--exploration", type=float, default=math.sqrt(2))
    parser.add_argument("--center-weight", type=float, default=0.5)
    parser.add_argument("--open-windows-weight", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows, summaries = run_experiments(
        strategies=args.strategies,
        runs=args.runs,
        games_per_side=args.games_per_side,
        seed=args.seed,
        output_dir=args.output_dir,
        depth=args.depth,
        simulations=args.simulations,
        iterations=args.iterations,
        exploration=args.exploration,
        center_weight=args.center_weight,
        open_windows_weight=args.open_windows_weight,
    )
    print(json.dumps({
        "runs_saved": len(rows),
        "summary": summaries,
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
