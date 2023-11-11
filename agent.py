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
import torch 


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
        """
        
        ENTER DOCSTRING - main agent trainer 
        
        """
        pass

    def retrain_nn(self, train_data): 
        """
        
        ENTER DOCSTRING - neural network trainer 
        
        """
        policy_loss_fn = torch.nn.CrossEntropyLoss()
        value_loss_fn = torch.nn.MSELoss()
        
        for state, policy, result in train_data: 
            x_train = state 
            policy_train = policy
            value_train = result
            
            policy_pred, value_pred = self.model.predict(state)

            policy_loss = policy_loss_fn(policy_train, policy_pred)
            value_loss = torch.nn.MSE(value_train, value_loss)


    def play_games(self, num_episodes: int, game: Game):
        """
        
        ENTER DOCSTRING - MCTS game player 
        
        """
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
                game_states.append((game_state, policy))
                action = np.random.choice(
                    np.arange(game.getActionSize()),
                    p=policy
                    )
                
                game_state, player = game_state.getNextState(
                    game_state,
                    player,
                    action)

            game_result = game_state.getGameEnded()
            for state, policy in game_states:
                train_episodes.append((state, policy, game_result))

        return train_episodes
