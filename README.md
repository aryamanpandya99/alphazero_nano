# AlphaZeroNano [in progress]

AlphaZeroNano is a miniature implementation of the famous AlphaZero, designed to understand and master grid-based board games. This project offers a lightweight framework for exploring the principles behind deep reinforcement learning combined with Monte Carlo Tree Search (MCTS) in the context of board games.

# Directory Structure

The project structure is as follows:

```
├── .github/workflows     # Contains GitHub workflow definitions (e.g., for linting with pylint)
├── Othello               # Added games from alpha_zero_general repository (MIT License)
├── TicTacToe             # Added games from alpha_zero_general repository (MIT License)
├── tests                 # Directory for test scripts
├── agent.py              # Implementation of AlphaZero agent and necessary functions: training & evaluation
├── main.py               # Main program entry
├── mcts.py               # Monte Carlo Tree Search and supporting functions
├── models.py             # Convolutional neural network for board state evaluation
└── README.md             # Documentation of the project

Please note that the project is structured to separate components logically, with the core functionality encapsulated in `agent.py`, `Game.py`, `mcts.py`, and `models.py`. The games Othello and TicTacToe from the alpha_zero_general repository are included under their respective directories, and tests are placed within the `tests` directory.
```

## Acknowledgments

Inspired by the groundbreaking work on AlphaZero by DeepMind, This project serves as an educational tool to understand the underlying principles and mechanisms in a simplified context
