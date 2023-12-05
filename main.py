"""
Author: Aryaman Pandya
File contents: Program entry point. Select game, initalize agent, train agent. 
This file will also be used to output plots and monitor training progress. 
This file will also be used for evaluation and testing. 
"""
import sys 
import logging
from agent import AlphaZeroNano
from models import OthelloNN
import torch

sys.path.append("Othello")
from othello_game import OthelloGame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(name)s | %(filename)s | %(funcName)s() | line.%(lineno)d | %(message)s"
)

def main():
    """
    main
    """
    neural_network = OthelloNN()
    game = OthelloGame(n=8)
    learning_rate = 3e-4
    l2_reg = 1e-3

    actor_optimizer = torch.optim.Adam(
        neural_network.parameters(),
        lr=learning_rate,
        weight_decay=l2_reg
    )

    agent = AlphaZeroNano(optimizer=actor_optimizer,num_simulations=5, game=game, c_uct=0.1)

    agent.train(
        train_batch_size=32,
        neural_network=neural_network,
        num_episodes=5,
        num_epochs=1000
        )


if __name__=="__main__":
    main()
