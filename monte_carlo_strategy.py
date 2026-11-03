"""Random-rollout Monte Carlo Connect Four strategy."""

from __future__ import annotations

import random
from typing import Optional

from connect_four_core import (
    COLUMNS,
    Board,
    _tactical_moves,
    _validate_search_request,
    drop_token,
    legal_moves,
    winner,
)


def _random_rollout(board: Board, to_move: int, root_player: int, rng: random.Random) -> float:
    # Roll out to a terminal state and score the result from the root's view.
    state = [list(row) for row in board]
    state_winner = winner(state)
    if state_winner is not None:
        return 1.0 if state_winner == root_player else 0.0
    while True:
        moves = legal_moves(state)
        if not moves:
            return 0.5
        column = rng.choice(moves)
        state = drop_token(state, column, to_move)
        state_winner = winner(state)
        if state_winner is not None:
            return 1.0 if state_winner == root_player else 0.0
        to_move = 3 - to_move


class MonteCarloStrategy:
    def __init__(self, simulations: int = 100, seed: Optional[int] = None) -> None:
        if type(simulations) is not int or simulations <= 0:
            raise ValueError("simulations must be a positive integer")
        self.simulations = simulations
        self._random = random.Random(seed)

    def __call__(self, board: Board, player: int) -> int:
        moves = _validate_search_request(board, player)
        tactical_moves = _tactical_moves(board, player, moves)
        if len(tactical_moves) != len(moves) or len(moves) == 1:
            return tactical_moves[0]

        results = {}
        for column in tactical_moves:
            next_board = drop_token(board, column, player)
            # Compare moves using the same fixed number of random continuations.
            total = sum(
                _random_rollout(next_board, 3 - player, player, self._random)
                for _ in range(self.simulations)
            )
            results[column] = total / self.simulations
        return max(
            tactical_moves,
            key=lambda column: (
                results[column],
                -abs(column - COLUMNS // 2),
                -column,
            ),
        )
