"""Weighted heuristic Connect Four strategy."""

from __future__ import annotations

import math
from dataclasses import dataclass

from connect_four_core import (
    COLUMNS,
    FOUR_WINDOWS,
    Board,
    _validate_board,
    _winning_moves,
    drop_token,
    legal_moves,
    winner,
)

MAX_WINDOW_SCORE = 69 * 16


@dataclass(frozen=True)
class HeuristicWeights:
    center: float
    open_windows: float

    def __post_init__(self) -> None:
        for name, value in (("center", self.center), ("open_windows", self.open_windows)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} weight must be a finite number from 0 to 1")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} weight must be a finite number from 0 to 1")


def _open_window_score(board: Board, player: int) -> float:
    # Reward lines with the player's tokens that contain no opponent tokens.
    opponent = 3 - player
    score = 0
    for window in FOUR_WINDOWS:
        values = [board[row][column] for row, column in window]
        if opponent not in values:
            own_tokens = values.count(player)
            if own_tokens:
                score += own_tokens * own_tokens
    return score / MAX_WINDOW_SCORE


def choose_move(board: Board, player: int, weights: HeuristicWeights) -> int:
    _validate_board(board)
    if type(player) is not int or player not in (1, 2):
        raise ValueError("player must be 1 or 2")
    if not isinstance(weights, HeuristicWeights):
        raise TypeError("weights must be a HeuristicWeights instance")
    if winner(board) is not None:
        raise ValueError("cannot choose a move after the game has been won")

    moves = legal_moves(board)
    if not moves:
        raise ValueError("cannot choose a move on a full board")

    winning_moves = _winning_moves(board, player)
    if winning_moves:
        return winning_moves[0]

    # Prefer a forced block before comparing heuristic feature scores.
    opponent_wins = set(_winning_moves(board, 3 - player))
    candidates = [column for column in moves if column in opponent_wins] or moves

    def score(column: int) -> tuple[float, float, int]:
        next_board = drop_token(board, column, player)
        center = 1 - abs(column - (COLUMNS // 2)) / (COLUMNS // 2)
        windows = _open_window_score(next_board, player)
        return (
            weights.center * center + weights.open_windows * windows,
            center,
            -column,
        )

    return max(candidates, key=score)
