"""Minimax Connect Four strategy with alpha-beta pruning."""

from __future__ import annotations

import math
from typing import Optional

from connect_four_core import (
    COLUMNS,
    FOUR_WINDOWS,
    ROWS,
    Board,
    _ordered_moves,
    _tactical_moves,
    _validate_search_request,
    drop_token,
    legal_moves,
    winner,
)
from heuristic_strategy import HeuristicWeights


def _static_evaluation(board: Board, player: int, weights: HeuristicWeights) -> float:
    # Score uncontested lines and center control from the root player's view.
    opponent = 3 - player
    pattern_weights = (0, 1, 8, 48, 0)
    window_value = 0
    for window in FOUR_WINDOWS:
        tokens = [board[row][column] for row, column in window]
        if player not in tokens:
            window_value -= pattern_weights[tokens.count(opponent)]
        elif opponent not in tokens:
            window_value += pattern_weights[tokens.count(player)]

    center_column = COLUMNS // 2
    center_value = 0
    for row in range(ROWS):
        if board[row][center_column] == player:
            center_value += 1
        elif board[row][center_column] == opponent:
            center_value -= 1
    return weights.center * center_value / ROWS + weights.open_windows * window_value / 500


class MinimaxStrategy:
    def __init__(
        self, depth: int = 4, weights: Optional[HeuristicWeights] = None
    ) -> None:
        if type(depth) is not int or depth <= 0:
            raise ValueError("depth must be a positive integer")
        self.depth = depth
        self.weights = weights or HeuristicWeights(0.5, 0.5)

    def __call__(self, board: Board, player: int) -> int:
        moves = _validate_search_request(board, player)
        tactical_moves = _tactical_moves(board, player, moves)
        if len(tactical_moves) != len(moves) or len(moves) == 1:
            return tactical_moves[0]

        def search(
            state: Board,
            to_move: int,
            remaining: int,
            alpha: float,
            beta: float,
        ) -> float:
            state_winner = winner(state)
            if state_winner == player:
                # Large terminal values dominate all heuristic leaf scores.
                return 1_000_000 + remaining
            if state_winner == 3 - player:
                return -1_000_000 - remaining
            if not legal_moves(state):
                return 0
            if remaining == 0:
                return _static_evaluation(state, player, self.weights)

            if to_move == player:
                value = -math.inf
                for column in _ordered_moves(state):
                    value = max(
                        value,
                        search(
                            drop_token(state, column, to_move),
                            3 - to_move,
                            remaining - 1,
                            alpha,
                            beta,
                        ),
                    )
                    alpha = max(alpha, value)
                    # Prune branches the minimizing player would never allow.
                    if alpha >= beta:
                        break
                return value

            value = math.inf
            for column in _ordered_moves(state):
                value = min(
                    value,
                    search(
                        drop_token(state, column, to_move),
                        3 - to_move,
                        remaining - 1,
                        alpha,
                        beta,
                    ),
                )
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

        best_column = tactical_moves[0]
        best_value = -math.inf
        alpha = -math.inf
        for column in _ordered_moves(board):
            if column not in tactical_moves:
                continue
            value = search(
                drop_token(board, column, player),
                3 - player,
                self.depth - 1,
                alpha,
                math.inf,
            )
            if value > best_value:
                best_value = value
                best_column = column
            alpha = max(alpha, best_value)
        return best_column
