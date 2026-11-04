"""Monte Carlo tree search Connect Four strategy using UCT."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
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
from monte_carlo_strategy import _random_rollout


@dataclass
class _MCTSNode:
    board: list[list[int]]
    player_to_move: int
    move: Optional[int]
    parent: Optional[_MCTSNode]
    untried_moves: list[int]
    children: list[_MCTSNode]
    visits: int = 0
    value_sum: float = 0.0


class MCTSStrategy:
    def __init__(
        self,
        iterations: int = 500,
        exploration: float = math.sqrt(2),
        seed: Optional[int] = None,
    ) -> None:
        if type(iterations) is not int or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if isinstance(exploration, bool) or not isinstance(exploration, (int, float)):
            raise ValueError("exploration must be a finite positive number")
        if not math.isfinite(exploration) or exploration <= 0:
            raise ValueError("exploration must be a finite positive number")
        self.iterations = iterations
        self.exploration = float(exploration)
        self._random = random.Random(seed)

    def __call__(self, board: Board, player: int) -> int:
        moves = _validate_search_request(board, player)
        tactical_moves = _tactical_moves(board, player, moves)
        if len(tactical_moves) != len(moves) or len(moves) == 1:
            return tactical_moves[0]

        root = _MCTSNode(
            board=[list(row) for row in board],
            player_to_move=player,
            move=None,
            parent=None,
            untried_moves=tactical_moves.copy(),
            children=[],
        )

        for _ in range(self.iterations):
            node = root
            # Descend by UCT until reaching a node with unexplored moves.
            while not node.untried_moves and node.children:
                maximizing = node.player_to_move == player

                def uct(child: _MCTSNode) -> float:
                    mean_value = child.value_sum / child.visits
                    exploitation = mean_value if maximizing else -mean_value
                    exploration = self.exploration * math.sqrt(
                        math.log(node.visits) / child.visits
                    )
                    return exploitation + exploration

                node = max(node.children, key=uct)

            if node.untried_moves:
                # Expand one untried move before simulating the resulting state.
                move_index = self._random.randrange(len(node.untried_moves))
                column = node.untried_moves.pop(move_index)
                child_board = drop_token(node.board, column, node.player_to_move)
                node = _MCTSNode(
                    board=child_board,
                    player_to_move=3 - node.player_to_move,
                    move=column,
                    parent=node,
                    untried_moves=[] if winner(child_board) else legal_moves(child_board),
                    children=[],
                )
                node.parent.children.append(node)

            reward = _random_rollout(node.board, node.player_to_move, player, self._random)
            # Store rewards from the root player's perspective throughout the path.
            while node is not None:
                node.visits += 1
                node.value_sum += reward
                node = node.parent

        def root_child_score(child: _MCTSNode) -> tuple[float, float, int, int]:
            if child.move is None:
                raise RuntimeError("MCTS root child has no move")
            return (
                child.visits,
                child.value_sum / child.visits,
                -abs(child.move - COLUMNS // 2),
                -child.move,
            )

        best_child = max(root.children, key=root_child_score)
        if best_child.move is None:
            raise RuntimeError("MCTS selected a root child without a move")
        return best_child.move
