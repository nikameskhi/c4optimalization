"""Tests for board rules, strategy behavior, and match evaluation."""

import unittest

from connect_four import evaluate_candidate, evaluate_strategy
from connect_four_core import (
    RandomLegalStrategy,
    drop_token,
    legal_moves,
    new_board,
    winner,
)
from heuristic_strategy import HeuristicWeights, choose_move
from mcts_strategy import MCTSStrategy
from minimax_strategy import MinimaxStrategy
from monte_carlo_strategy import MonteCarloStrategy


class ConnectFourTests(unittest.TestCase):
    """Verify legal play and expected strategy decisions on small positions."""

    def test_move_lands_in_lowest_empty_cell(self):
        board = drop_token(new_board(), 3, 1)
        board = drop_token(board, 3, 2)
        self.assertEqual(board[5][3], 1)
        self.assertEqual(board[4][3], 2)
        self.assertEqual(board[3][3], 0)

    def test_full_column_is_not_legal(self):
        board = new_board()
        for player in (1, 2, 1, 2, 1, 2):
            board = drop_token(board, 0, player)
        self.assertNotIn(0, legal_moves(board))
        with self.assertRaisesRegex(ValueError, "is full"):
            drop_token(board, 0, 1)

    def test_detects_horizontal_win(self):
        board = new_board()
        for column in range(4):
            board = drop_token(board, column, 1)
        self.assertEqual(winner(board), 1)

    def test_heuristic_takes_immediate_win(self):
        board = new_board()
        for player, column in ((1, 0), (2, 6), (1, 1), (2, 6), (1, 2), (2, 5)):
            board = drop_token(board, column, player)
        self.assertEqual(choose_move(board, 1, HeuristicWeights(0, 0)), 3)

    def test_heuristic_blocks_immediate_loss(self):
        board = new_board()
        for player, column in ((1, 6), (2, 0), (1, 5), (2, 1), (1, 4), (2, 2)):
            board = drop_token(board, column, player)
        self.assertEqual(choose_move(board, 1, HeuristicWeights(0, 0)), 3)

    def test_search_strategies_take_immediate_win_and_block(self):
        strategies = (
            MinimaxStrategy(depth=2),
            MonteCarloStrategy(simulations=2, seed=3),
            MCTSStrategy(iterations=10, seed=3),
        )
        winning_board = new_board()
        blocking_board = new_board()
        for player, column in ((1, 0), (2, 6), (1, 1), (2, 6), (1, 2), (2, 5)):
            winning_board = drop_token(winning_board, column, player)
        for player, column in ((1, 6), (2, 0), (1, 5), (2, 1), (1, 4), (2, 2)):
            blocking_board = drop_token(blocking_board, column, player)

        for strategy in strategies:
            with self.subTest(strategy=type(strategy).__name__):
                self.assertEqual(strategy(winning_board, 1), 3)
                self.assertEqual(strategy(blocking_board, 1), 3)

    def test_search_strategies_return_legal_moves(self):
        board = new_board()
        strategies = (
            MinimaxStrategy(depth=2),
            MonteCarloStrategy(simulations=2, seed=7),
            MCTSStrategy(iterations=10, seed=7),
        )
        for strategy in strategies:
            with self.subTest(strategy=type(strategy).__name__):
                self.assertIn(strategy(board, 1), legal_moves(board))

    def test_monte_carlo_strategies_are_reproducible_with_same_seed(self):
        board = new_board()
        first = MonteCarloStrategy(simulations=5, seed=19)
        second = MonteCarloStrategy(simulations=5, seed=19)
        self.assertEqual(first(board, 1), second(board, 1))

    def test_mcts_strategies_are_reproducible_with_same_seed(self):
        board = new_board()
        first = MCTSStrategy(iterations=20, seed=19)
        second = MCTSStrategy(iterations=20, seed=19)
        self.assertEqual(first(board, 1), second(board, 1))

    def test_weight_constraints_are_enforced(self):
        for value in (-0.1, 1.1, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                HeuristicWeights(value, 0.5)

    def test_search_budgets_are_validated(self):
        for create_strategy in (
            lambda: MinimaxStrategy(depth=0),
            lambda: MonteCarloStrategy(simulations=0),
            lambda: MCTSStrategy(iterations=0),
            lambda: MCTSStrategy(iterations=1, exploration=float("nan")),
        ):
            with self.subTest(create_strategy=create_strategy), self.assertRaises(ValueError):
                create_strategy()

    def test_random_strategy_returns_legal_move(self):
        board = new_board()
        move = RandomLegalStrategy(seed=7)(board, 1)
        self.assertIn(move, legal_moves(board))

    def test_evaluation_uses_both_player_positions(self):
        results = evaluate_candidate(HeuristicWeights(0.5, 0.5), games_per_side=1, seed=11)
        self.assertEqual(results.games, 2)
        self.assertEqual(results.first_player_games, 1)
        self.assertEqual(results.second_player_games, 1)
        self.assertEqual(results.started_first_games, 1)
        self.assertEqual(results.started_second_games, 1)
        self.assertEqual(results.wins + results.draws + results.losses, results.games)
        self.assertGreaterEqual(results.score, 0)
        self.assertLessEqual(results.score, 1)

    def test_evaluation_accepts_other_strategy_factories(self):
        results = evaluate_strategy(
            lambda seed: MCTSStrategy(iterations=5, seed=seed),
            games_per_side=1,
            seed=11,
        )
        self.assertEqual(results.games, 2)
        self.assertEqual(results.first_player_games, 1)
        self.assertEqual(results.second_player_games, 1)
        self.assertEqual(results.started_first_games, 1)
        self.assertEqual(results.started_second_games, 1)


if __name__ == "__main__":
    unittest.main()
