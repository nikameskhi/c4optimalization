# Connect Four

This project studies how to design and optimize complete playing strategies for Connect Four. A candidate solution is a set of rules or parameters used to choose moves throughout a game, rather than a single move. Strategies will be evaluated against multiple opponents and from both starting positions.

## Problem analysis

Connect Four is a two-player, turn-based, perfect-information game played on a 6-row by 7-column board. On each turn, a player drops a token into a non-full column, where it occupies the lowest available cell. A player wins by connecting four tokens horizontally, vertically, or diagonally; a full board without a winner is a draw.

At a decision point, the strategy receives the board state, the player to move, and the legal moves. Whole-strategy evaluation also depends on the selected opponents, match rules, and experiment configuration. The main formulation is to find strategy parameters $\theta$ that maximize average match score:

$$
F(\theta) = \frac{1}{N}\sum_{i=1}^{N} r_i
$$

where each game result $r_i$ is 1 for a win, 0.5 for a draw, and 0 for a loss. Candidate evaluation uses a fixed opponent set and conditions; parameters should be tuned on training games and final results measured on a separate test set.

## Problem classification

- **Numerical, mixed-variable optimization:** heuristic weights are real-valued, while search depth and simulation budgets are integer-valued.
- **Combined and constrained:** the goal is to maximize strategy performance while respecting parameter domains, legal-move rules, and computational limits.
- **Single-objective:** average game score is the objective; computational cost is treated as a constraint in the primary formulation.
- **Deterministic or stochastic evaluation:** game rules and deterministic strategies such as Minimax are deterministic under fixed tie-breaking; Monte Carlo methods introduce randomness through simulations.
- **Static optimization:** the opponent set and evaluation conditions are fixed while optimizing a candidate, even though game states change during play.
- **Black-box with some grey-box information:** candidate quality is observed by playing games, while the game rules and heuristic features are known.

## Candidate strategy representations

The analysis considers several complete strategy designs:

- **Parameterized heuristic:** score legal moves with a weighted function, for example using immediate wins, blocking, center control, threats, and mobility; choose the highest-scoring move.
- **Minimax with alpha-beta pruning:** search to a configured depth and evaluate non-terminal leaf states with a heuristic.
- **Monte Carlo simulations:** estimate each move's value from simulated game continuations.
- **Monte Carlo Tree Search (MCTS):** allocate simulations selectively across a search tree, for example using UCB1.

The board can be represented as a 6×7 matrix containing empty cells and the two players' tokens; a bitboard is an alternative for more efficient operations. All strategies should share the same legal-move generation, move application, and game-end checks.

## Constraints and evaluation

Strategies must select a legal move whenever the game is ongoing, never play in a full column, and stop after a win or draw. Candidate parameters and each method's search or simulation budget must remain within defined bounds. Invalid moves should be recorded as errors or handled using a documented penalty or recovery rule, not silently accepted.

Experiments should use the same opponents, rules, and comparable computational budgets across methods. Each strategy should play as both the first and second player; stochastic methods should be tested with multiple random seeds. Report wins, draws, losses, average score, and relevant costs such as decision time, searched nodes, or simulations.

## Alternative formulations

Possible variants include discrete rather than continuous weights, rule-based strategies, multi-objective optimization of score and decision cost, strategies that adapt during play, and combinations of heuristic evaluation with search. These alternatives change the candidate representation, optimization space, or constraints and should be justified before implementation.

## Project contents

- `analysis.pdf` – detailed analysis of the problem and proposed experiment plan.

## Running the project

Run instructions will be added once the implementation and its dependencies are in place.
