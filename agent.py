"""
Author: Aryaman Pandya
File contents: Implementation of our AlphaZero Nano agent.
This file contains the training code and supporting functions
required to enable training through self play.
"""
import sys

import numpy as np

sys.path.append("Othello")

from othello_game import OthelloGame

from mcts import apv_mcts
from models import OthelloNN
from Game import Game


class AlphaZeroNano:
    """
    Class implementation of AlphaZero agent.
    """

    def __init__(
            self,
            neural_network: OthelloNN,
            num_simulations: int,
            C: float) -> None:

        self.model = neural_network
        self.c_parameter = C
        self.num_simulations = num_simulations

    def train(self):
        # note to self -> use notes to write out trainer
        pass

    def play_games(self, num_episodes: int, game: Game):

        train_episodes = []
        player = 1
        for _ in range(num_episodes):
            game_states = []
            game_state = game.getInitBoard()

            while not game_state.getGameEnded():
                policy = apv_mcts(
                    game=game,
                    root_state=game_state,
                    model=self.model,
                    num_iterations=self.num_simulations,
                    c=self.c_parameter,
                )
                game_states.append((game, policy))
                action = np.random.choice(
                    np.arange(game.getActionSize()),
                    p=policy
                    )
                # need to figure out how to handle player expression
                game_state, player = game_state.getNextState(
                    game_state,
                    player,
                    action)

            game_result = game_state.getScore()
            for state, policy in game_states:
                train_episodes.append((state, policy, game_result))

        return train_episodes
