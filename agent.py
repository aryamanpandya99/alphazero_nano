"""
Author: Aryaman Pandya
File contents: Implementation of our AlphaZero Nano agent.
This file contains the training code and supporting functions
required to enable training through self play.
"""
import numpy as np
import copy
import torch
from torch.utils.data import DataLoader, TensorDataset

from Game import Game
from mcts import apv_mcts
from models import OthelloNN


class AlphaZeroNano:
    """
    Class implementation of AlphaZero agent.
    """

    def __init__(
            self,
            num_simulations: int,
            optimizer: torch.optim,
            learning_rate: float,
            regularization: float,
            C: float) -> None:

        self.c_parameter = C
        self.num_simulations = num_simulations

        self.optimizer = optimizer(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=regularization
            )

    def train(self, neural_network: torch.nn.Module, train_batch_size: int, num_epochs: int, num_episodes: int):
        """
while TRAINING_NOT_CONVERGED:
    training_data = SELF_PLAY(current_network)
    RETRAIN_NETWORK(current_network, training_data)
    new_network = COPY_OF(current_network)
    best_network = EVALUATE_NETWORK(new_network, current_network)
    if best_network == new_network:
        current_network = new_network
        """
        current_network = neural_network
        
        for i in range(num_epochs):
            train_episodes = self.self_play()

            # keep a copy of the current network for evaluation 
            old_network = copy.deepcopy(current_network)
            self.retrain_nn(
                neural_network=current_network,
                train_data=train_episodes,
                train_batch_size=train_batch_size
                )
        pass

    def evaluate_networks(self, current_network: torch.nn.Module, new_network: torch.nn.Module, num_games: int): 
        
        new_network_wins = 0 

        for _ in range(num_games):
            new_network_won = self.play_games(current_network, new_network)
            
            if new_network_won: 
                new_network_wins+=1

    def retrain_nn(self, neural_network: torch.nn.Module, train_data: list, train_batch_size: int) -> None:
        """
        ENTER DOCSTRING - neural network trainer
        """
        policy_loss_fn = torch.nn.CrossEntropyLoss()
        value_loss_fn = torch.nn.MSELoss()

        dataloader = self.batch_episodes(
            train_data, batch_size=train_batch_size
            )

        for x_train, policy_train, value_train in dataloader:
            policy_pred, value_pred = neural_network.predict(x_train)

            policy_loss = policy_loss_fn(policy_train, policy_pred)
            value_loss = value_loss_fn(value_train, value_pred)
            combined_loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            combined_loss.backward()
            self.optimizer.step()
        


    def self_play(self, num_episodes: int, game: Game):
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

    def batch_episodes(train_data: list, batch_size: int):
        """
        ENTER DOCSTRING - tensor batching helper
        """
        states, policies, results = zip(*train_data)
        states_tensor = torch.tensor(states, dtype=torch.float32)
        policies_tensor = torch.tensor(policies, dtype=torch.float32)
        results_tensor = torch.tensor(results, dtype=torch.float32)

        dataset = TensorDataset(states_tensor, policies_tensor, results_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader
