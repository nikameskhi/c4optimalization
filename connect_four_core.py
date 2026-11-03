"""Shared Connect Four rules and game-running utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

ROWS = 6
COLUMNS = 7
Board = Sequence[Sequence[int]]
Strategy = Callable[[Board, int], int]


def _four_windows() -> tuple[tuple[tuple[int, int], ...], ...]:
    # Precompute every horizontal, vertical, and diagonal four-cell line.
    windows = []
    for row in range(ROWS):
        for column in range(COLUMNS - 3):
            windows.append(tuple((row, column + offset) for offset in range(4)))
    for row in range(ROWS - 3):
        for column in range(COLUMNS):
            windows.append(tuple((row + offset, column) for offset in range(4)))
    for row in range(ROWS - 3):
        for column in range(COLUMNS - 3):
            windows.append(tuple((row + offset, column + offset) for offset in range(4)))
        for column in range(3, COLUMNS):
            windows.append(tuple((row + offset, column - offset) for offset in range(4)))
    return tuple(windows)


FOUR_WINDOWS = _four_windows()


def _validate_board(board: Board) -> None:
    if len(board) != ROWS or any(len(row) != COLUMNS for row in board):
        raise ValueError(f"board must have {ROWS} rows and {COLUMNS} columns")
    if any(type(cell) is not int or cell not in (0, 1, 2) for row in board for cell in row):
        raise ValueError("board cells must be integers: 0 (empty), 1, or 2")


def new_board() -> list[list[int]]:
    return [[0 for _ in range(COLUMNS)] for _ in range(ROWS)]


def legal_moves(board: Board) -> list[int]:
    _validate_board(board)
    # A column is playable exactly when its top cell is empty.
    return [column for column in range(COLUMNS) if board[0][column] == 0]


def drop_token(board: Board, column: int, player: int) -> list[list[int]]:
    _validate_board(board)
    if type(column) is not int or column < 0 or column >= COLUMNS:
        raise ValueError(f"column must be an integer from 0 to {COLUMNS - 1}")
    if type(player) is not int or player not in (1, 2):
        raise ValueError("player must be 1 or 2")
    if board[0][column] != 0:
        raise ValueError(f"column {column} is full")

    updated = [list(row) for row in board]
    for row in range(ROWS - 1, -1, -1):
        if updated[row][column] == 0:
            updated[row][column] = player
            return updated
    raise ValueError(f"column {column} is full")


def winner(board: Board) -> Optional[int]:
    _validate_board(board)
    winners = {
        board[window[0][0]][window[0][1]]
        for window in FOUR_WINDOWS
        if board[window[0][0]][window[0][1]] != 0
        and all(
            board[row][column] == board[window[0][0]][window[0][1]]
            for row, column in window[1:]
        )
    }
    if len(winners) > 1:
        raise ValueError("board cannot contain winning lines for both players")
    return next(iter(winners), None)


def is_draw(board: Board) -> bool:
    return winner(board) is None and not legal_moves(board)


def _winning_moves(board: Board, player: int) -> list[int]:
    return [
        column
        for column in legal_moves(board)
        if winner(drop_token(board, column, player)) == player
    ]


def _validate_search_request(board: Board, player: int) -> list[int]:
    _validate_board(board)
    if type(player) is not int or player not in (1, 2):
        raise ValueError("player must be 1 or 2")
    if winner(board) is not None:
        raise ValueError("cannot choose a move after the game has been won")
    moves = legal_moves(board)
    if not moves:
        raise ValueError("cannot choose a move on a full board")
    return moves


def _tactical_moves(board: Board, player: int, moves: list[int]) -> list[int]:
    winning_moves = _winning_moves(board, player)
    if winning_moves:
        return winning_moves
    opponent_wins = _winning_moves(board, 3 - player)
    # A single immediate threat can be blocked; multiple threats need deeper search.
    if len(opponent_wins) != 1:
        return moves
    blocking_moves = [column for column in moves if column == opponent_wins[0]]
    return blocking_moves or moves


def _ordered_moves(board: Board) -> list[int]:
    return sorted(legal_moves(board), key=lambda column: (abs(column - COLUMNS // 2), column))


class RandomLegalStrategy:
    def __init__(self, seed: Optional[int] = None) -> None:
        self._random = random.Random(seed)

    def __call__(self, board: Board, player: int) -> int:
        moves = legal_moves(board)
        if not moves:
            raise ValueError("random strategy cannot move on a full board")
        return self._random.choice(moves)


@dataclass(frozen=True)
class GameResult:
    winner: Optional[int]
    moves: int


def play_game(player_one: Strategy, player_two: Strategy, starting_player: int = 1) -> GameResult:
    if type(starting_player) is not int or starting_player not in (1, 2):
        raise ValueError("starting_player must be 1 or 2")
    board = new_board()
    player = starting_player

    while True:
        strategy = player_one if player == 1 else player_two
        # Give strategies a snapshot so they cannot mutate the game's board.
        column = strategy([row.copy() for row in board], player)
        board = drop_token(board, column, player)
        moves = ROWS * COLUMNS - sum(cell == 0 for row in board for cell in row)

        if winner(board) == player:
            return GameResult(winner=player, moves=moves)
        if not legal_moves(board):
            return GameResult(winner=None, moves=moves)
        player = 3 - player
