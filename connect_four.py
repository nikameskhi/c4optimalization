"""Evaluate Connect Four strategies against the random-legal-move baseline."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Callable

from connect_four_core import (
    Board,
    COLUMNS,
    RandomLegalStrategy,
    ROWS,
    Strategy,
    drop_token,
    is_draw,
    legal_moves,
    new_board,
    play_game,
    winner,
)
from heuristic_strategy import HeuristicWeights, choose_move
from mcts_strategy import MCTSStrategy
from minimax_strategy import MinimaxStrategy
from monte_carlo_strategy import MonteCarloStrategy


@dataclass(frozen=True)
class EvaluationResult:
    wins: int
    draws: int
    losses: int
    games: int
    score: float
    first_player_games: int
    second_player_games: int
    started_first_games: int
    started_second_games: int


def evaluate_strategy(
    strategy_factory: Callable[[int], Strategy],
    games_per_side: int = 10,
    seed: int = 0,
) -> EvaluationResult:
    if type(games_per_side) is not int or games_per_side <= 0:
        raise ValueError("games_per_side must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    wins = draws = losses = 0
    started_first_games = started_second_games = 0
    for candidate_seat in (1, 2):
        for game_number in range(games_per_side):
            game_seed = seed + game_number
            # Alternate turn order while testing the candidate in both board seats.
            starting_player = 1 if game_number % 2 == 0 else 2
            baseline = RandomLegalStrategy(game_seed)
            candidate = strategy_factory(game_seed + 1_000_003)
            strategies = (candidate, baseline) if candidate_seat == 1 else (baseline, candidate)
            result = play_game(*strategies, starting_player=starting_player)
            if candidate_seat == starting_player:
                started_first_games += 1
            else:
                started_second_games += 1
            if result.winner is None:
                draws += 1
            elif result.winner == candidate_seat:
                wins += 1
            else:
                losses += 1

    games = 2 * games_per_side
    # Standard match score: win = 1, draw = 0.5, loss = 0.
    score = (wins + 0.5 * draws) / games
    return EvaluationResult(
        wins=wins,
        draws=draws,
        losses=losses,
        games=games,
        score=score,
        first_player_games=games_per_side,
        second_player_games=games_per_side,
        started_first_games=started_first_games,
        started_second_games=started_second_games,
    )


def evaluate_candidate(
    weights: HeuristicWeights, games_per_side: int = 10, seed: int = 0
) -> EvaluationResult:
    return evaluate_strategy(
        lambda _seed: lambda board, player: choose_move(board, player, weights),
        games_per_side,
        seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=("heuristic", "minimax", "monte-carlo", "mcts"),
        default="heuristic",
    )
    parser.add_argument("--center-weight", type=float, default=0.5)
    parser.add_argument("--open-windows-weight", type=float, default=0.5)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--exploration", type=float, default=math.sqrt(2))
    parser.add_argument("--games-per-side", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    weights = HeuristicWeights(args.center_weight, args.open_windows_weight)
    if args.strategy == "heuristic":
        strategy_factory = lambda _seed: (
            lambda board, player: choose_move(board, player, weights)
        )
        parameters = {"weights": asdict(weights)}
    elif args.strategy == "minimax":
        strategy_factory = lambda _seed: MinimaxStrategy(args.depth, weights)
        parameters = {"depth": args.depth, "weights": asdict(weights)}
    elif args.strategy == "monte-carlo":
        strategy_factory = lambda game_seed: MonteCarloStrategy(args.simulations, game_seed)
        parameters = {"simulations_per_move": args.simulations}
    else:
        strategy_factory = lambda game_seed: MCTSStrategy(
            args.iterations, args.exploration, game_seed
        )
        parameters = {
            "iterations_per_move": args.iterations,
            "exploration": args.exploration,
        }

    results = evaluate_strategy(strategy_factory, args.games_per_side, args.seed)
    print(json.dumps({
        "strategy": args.strategy,
        "parameters": parameters,
        "baseline": "random_legal",
        "seed": args.seed,
        "results": asdict(results),
    }, indent=2))


if __name__ == "__main__":
    main()
