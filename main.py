"""
Author: Aryaman Pandya
File contents: Program entry point. Select game, initalize agent, train agent. 
This file will also be used to output plots and monitor training progress. 
This file will also be used for evaluation and testing. 
"""

import logging
import sys

import torch

from agent import AlphaZeroNano
from mcts import MCTS
from models import OthelloNN

sys.path.append("Othello")
from othello_game import OthelloGame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(name)s | %(filename)s | %(funcName)s() | line.%(lineno)d | %(message)s",
)


def main():
    """
    main
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    neural_network = OthelloNN()
    neural_network = neural_network.to(device)
    game = OthelloGame(n=8)
    learning_rate = 3e-4
    l2_reg = 1e-3

    actor_optimizer = torch.optim.Adam(
        neural_network.parameters(), lr=learning_rate, weight_decay=l2_reg
    )
    agent = AlphaZeroNano(
        optimizer=actor_optimizer,
        num_simulations=25,
        game=game,
        c_uct=1,
        device=device,
        mcts=MCTS(game),
    )

    agent.train(
        train_batch_size=16,
        neural_network=neural_network,
        num_episodes=100,
        num_epochs=100,
    )


if __name__ == "__main__":
    main()
