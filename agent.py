"""
Author: Aryaman Pandya
File contents: Implementation of our AlphaZero Nano agent.
This file contains the training code and supporting functions
required to enable training through self play.
"""
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from Game import Game
from mcts import apv_mcts


class AlphaZeroNano:
    """
    Class implementation of AlphaZero agent.
    """

    def __init__(
            self,
            num_simulations: int,
            optimizer,
            game: Game,
            c_uct: float) -> None:

        self.c_parameter = c_uct
        self.num_simulations = num_simulations
        self.optimizer = optimizer
        self.game = game

    def train(self,
              neural_network: torch.nn.Module,
              train_batch_size: int,
              num_epochs: int,
              num_episodes: int):
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

        for _ in range(num_epochs):
            train_episodes = self.self_play(
                model=current_network,
                num_episodes=num_episodes
            )
            # keep a copy of the current network for evaluation
            old_network = copy.deepcopy(current_network)

            self.retrain_nn(
                neural_network=current_network,
                train_data=train_episodes,
                train_batch_size=train_batch_size
            )
            # note: figure out if the following assignment makes sense
            current_network = self.evaluate_networks(
                current_network,
                old_network,
                10
            )

    def evaluate_networks(self,
                          network_a: torch.nn.Module,
                          network_b: torch.nn.Module,
                          num_games: int,
                          threshold=0.6):
        """
        ENTER DOCSTRING - neural network trainer
        """
        network_a_wins = 0

        for _ in range(num_games):
            network_a_result = self.play_game(network_a, network_b)
            if network_a_result > 0:
                network_a_wins += 1

        win_rate = network_a_wins / num_games

        if win_rate > threshold:
            return network_a

        return network_b

    def retrain_nn(self,
                   neural_network: torch.nn.Module,
                   train_data: list,
                   train_batch_size: int) -> None:
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

    def play_game(self,
                  network_a: torch.nn.Module,
                  network_b: torch.nn.Module) -> bool:
        """
        ENTER DOCSTRING - MCTS game player
        """
        game_state = self.game.getInitBoard()
        player = 1

        while not game_state.getGameEnded():
            if player == 1:
                policy, _ = network_a.predict(game_state)
            else:
                policy, _ = network_b.predict(game_state)

            _, action = torch.max(policy, dim=-1)

            game_state, player = game_state.getNextState(
                game_state,
                player,
                action
            )

        if player == -1:
            return game_state.getGameEnded()

        return game_state.getGameEnded()

    def self_play(self, model: torch.nn.Model, num_episodes: int):
        """
        ENTER DOCSTRING - MCTS game player
        """
        train_episodes = []

        for _ in range(num_episodes):
            game_states = []
            game_state = self.game.getInitBoard()
            player = 1
            while not game_state.getGameEnded():
                policy = apv_mcts(
                    game=self.game,
                    root_state=game_state,
                    model=model,
                    num_iterations=self.num_simulations,
                    c=self.c_parameter,
                )

                game_states.append((game_state, policy))

                action = np.random.choice(
                    np.arange(self.game.getActionSize()),
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

    def batch_episodes(self, train_data: list, batch_size: int):
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
