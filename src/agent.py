"""
Author: Aryaman Pandya
File contents: Implementation of our AlphaZero Nano agent.
This file contains the training code and supporting functions
required to enable training through self play.
"""

import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output
from torch.utils.data import DataLoader, TensorDataset

from src.mcts import MCTS


class AlphaZeroAgent:
    """
    Class implementation of AlphaZero agent.
    """

    def __init__(
        self,
        num_simulations: int,
        optimizer,
        game,
        c_uct: float,
        mcts: MCTS,
        device,
    ) -> None:

        self.c_parameter = c_uct
        self.num_simulations = num_simulations
        self.optimizer = optimizer
        self.game = game
        self.device = device
        self.mcts = mcts
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(5, 5))
        plt.ion()

    def train(
        self,
        neural_network: torch.nn.Module,
        train_batch_size: int,
        num_epochs: int,
        num_episodes: int,
    ) -> torch.nn.Module:
        """
        Main agent trainer. Combines self play with MCTS and
        network retraining based on this experience to develop
        our agent. Returns trained neural network.

        Args:
            neural_network (torch.nn.Module)
            train_batch_size (int)
            num_epochs (int)
            num_episodes (int)

        Returns:
            current_network (torch.nn.Module)
        """
        logging.info("Beginning agent AlphaZeroNano training")
        current_network = neural_network

        policy_losses = []
        value_losses = []

        for epoch in range(num_epochs):
            train_episodes = self.self_play(
                model=current_network, num_episodes=num_episodes
            )
            old_network = copy.deepcopy(current_network)
            policy_loss, value_loss = self.retrain_nn(
                neural_network=current_network,
                train_data=train_episodes,
                train_batch_size=train_batch_size,
            )
            current_network = self.evaluate_networks(current_network, old_network, 10)

            policy_losses.append(policy_loss)
            value_losses.append(value_loss)

            self.plot_losses(policy_losses, value_losses)

            logging.info(
                "Epoch: %s/%s value_loss: %s, policy_loss: %s",
                epoch + 1,
                num_epochs,
                value_loss,
                policy_loss,
            )

        plt.ioff()
        plt.show()

        return current_network

    def plot_losses(self, policy_losses, value_losses):
        """
        Update the plot with the latest policy and value losses.

        Args:
            policy_losses (list): List of policy losses
            value_losses (list): List of value losses
        """
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(policy_losses, label="Policy Loss")
        self.ax1.set_title("Policy Loss over Epochs")
        self.ax1.set_xlabel("Epoch")
        self.ax1.set_ylabel("Loss")
        self.ax1.legend()

        self.ax2.plot(value_losses, label="Value Loss")
        self.ax2.set_title("Value Loss over Epochs")
        self.ax2.set_xlabel("Epoch")
        self.ax2.set_ylabel("Loss")
        self.ax2.legend()

        self.fig.tight_layout()
        plt.draw()
        plt.pause(0.1)
        clear_output(wait=True)
        self.fig.canvas.draw()

    def evaluate_networks(
        self,
        curr_network: torch.nn.Module,
        network_b: torch.nn.Module,
        num_games: int,
        threshold=0.5,
    ) -> torch.nn.Module:
        """
        Evaluate networks. Makes two networks play a specified
        number of games against one another and returns the second
        network if it beats the first beyond some threshold %

        Args:
            curr_network (torch.nn.Module)
            network_b (torch.nn.Module)
            num_games (int)
            threshold (float)

        Returns:
            network (torch.nn.Module)
        """
        curr_network_wins = 0

        for _ in range(num_games):
            print(f"game number: {_}")
            curr_network_result = self.play_game(curr_network, network_b)
            if curr_network_result > 0:
                curr_network_wins += 1

        win_rate = curr_network_wins / num_games
        logging.info(f"Current network win rate: {win_rate:.2f}")

        if win_rate > threshold:
            return curr_network

        # else return network b
        print(f"new network wins with win rate: {1 - win_rate}")
        return network_b

    def retrain_nn(
        self, neural_network: torch.nn.Module, train_data: list, train_batch_size: int
    ) -> None:
        """
        Neural network trainer function.

        Args:
            neural_network (torch.nn.Module)
            train_data (list)
            train_batch_size (int)

        Returns:
            N/A
        """
        policy_loss_fn = torch.nn.CrossEntropyLoss()
        value_loss_fn = torch.nn.MSELoss()

        dataloader = self.batch_episodes(train_data, batch_size=train_batch_size)
        policy_losses_total = 0
        value_losses_total = 0

        for x_train, policy_train, value_train in dataloader:
            policy_pred, value_pred = neural_network(x_train)
            policy_loss = policy_loss_fn(policy_pred, policy_train)
            value_loss = value_loss_fn(value_pred, value_train)

            policy_losses_total += policy_loss.item()
            value_losses_total += value_loss.item()
            combined_loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            combined_loss.backward()
            self.optimizer.step()

        policy_loss_avg = policy_losses_total / len(dataloader)
        value_losses_avg = value_losses_total / len(dataloader)

        return policy_loss_avg, value_losses_avg

    def play_game(self, network_a: torch.nn.Module, network_b: torch.nn.Module) -> bool:
        """
        Makes two network play a game against one another.
        Args:
            network_a (torch.nn.Module)
            network_b (torch.nn.Module)
        Returns:
            result (bool)
        """
        game_state = self.game.getInitBoard()
        player = np.random.choice([-1, 1])
        print(f"starting player: {player}")
        game_state = self.game.getCanonicalForm(game_state, player)
        stacked_frames = self.mcts.no_history_model_input(
            game_state, current_player=player
        )
        while not self.game.getGameEnded(board=game_state, player=player):
            # print(f"stacked frames: \n{stacked_frames}")
            stacked_tensor = (
                torch.tensor(stacked_frames, dtype=torch.float32)
                .to(self.device)
                .unsqueeze(0)
            )
            print(f"Player {player}, state: \n{game_state}")
            if player == 1:
                policy, _ = network_a(stacked_tensor)

            else:
                policy, _ = network_b(stacked_tensor)

            valid_moves = self.game.getValidMoves(game_state, player)
            ones_indices = np.where(valid_moves == 1)[0]
            mask = torch.zeros_like(policy.squeeze(), dtype=torch.bool)
            mask[torch.tensor(ones_indices)] = True

            policy[~mask] = 0
            action = torch.argmax(policy, dim=-1)

            game_state, player = self.game.getNextState(game_state, player, action)
            game_state = self.game.getCanonicalForm(game_state, player)
            stacked_frames = self.mcts.no_history_model_input(
                game_state, current_player=player
            )

        if player == -1:
            print(
                f"player: {player}, result: {self.game.getGameEnded(game_state, player)}"
            )
            return -self.game.getGameEnded(board=game_state, player=player)
        print(f"player: {player}, result: {self.game.getGameEnded(game_state, player)}")
        return self.game.getGameEnded(board=game_state, player=player)

    def self_play(self, model: torch.nn.Module, num_episodes: int):
        """
        Self play. Simulates number of episodes using MCTS and a given
        neural network.

        Args:
            model (torch.nn.Module)
            num_episodes (int)

        Returns:
            episodes (list)
        """
        train_episodes = []

        for _ in range(num_episodes):
            self.mcts = MCTS(self.game)
            game_states = []
            game_state = self.game.getInitBoard()
            player = 1
            game_state = self.game.getCanonicalForm(game_state, player=player)
            while not self.game.getGameEnded(board=game_state, player=player):
                pi = self.mcts.apv_mcts(
                    canonical_root_state=game_state,
                    model=model,
                    num_iterations=self.num_simulations,
                    uct_c=self.c_parameter,
                    device=self.device,
                )
                game_states.append((game_state, pi, player))
                valid_moves = self.game.getValidMoves(game_state, player)
                sum_pi = np.sum(pi)

                if sum_pi > 1e-8:
                    pi = pi / sum_pi
                else:
                    pi = valid_moves.astype(float) / np.sum(valid_moves)

                action = np.random.choice(len(pi), p=pi)
                game_state, player = self.game.getNextState(game_state, player, action)
                game_state = self.game.getCanonicalForm(game_state, player=player)

            game_result = self.game.getGameEnded(board=game_state, player=player)
            last_player = player

            for state, pi, player in game_states:
                stacked_frames = self.mcts.no_history_model_input(
                    state, current_player=player
                )
                if player != last_player:
                    train_episodes.append((stacked_frames, pi, -game_result))
                train_episodes.append((stacked_frames, pi, game_result))

        return train_episodes

    def batch_episodes(self, train_data: list, batch_size: int):
        """
        Batch episodes: given a list of experience, unpacks and
        converts to torch loaders.

        Args:
            train_data (list)
            batch_size (int)

        Returns:
            dataloader (TensorDataset)

        Note for future optimization (once debugged):
        UserWarning:
        Creating a tensor from a list of numpy.ndarrays is extremely slow.
        Please consider converting the list to a single numpy.ndarray with numpy.array()
        before converting to a tensor.
        """
        states, policies, results = zip(*train_data)
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(
            self.device
        )
        policies_tensor = torch.tensor(np.array(policies), dtype=torch.float32).to(
            self.device
        )
        results_tensor = torch.tensor(np.array(results), dtype=torch.float32).to(
            self.device
        )

        dataset = TensorDataset(states_tensor, policies_tensor, results_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader
