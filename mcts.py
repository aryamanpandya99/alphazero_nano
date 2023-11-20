"""
Author: Aryaman Pandya
File contents: Implementation of Monte Carlo Tree Search as well and the
components needed to support it
"""

import logging
import random

import numpy as np
import torch

from Game import Game


class Node:
    """
    Class: MCTS node 
    """
    def __init__(self, state, action_space: int) -> None:

        self.state = state
        self.children = {}

        """
        the following are the quantities required to be stored as per the
        MCTS implementation in the AlphaGo Zero paper, extended in the
        AlphaZero paper
        """

        self.total_value_s_a = [0 for _ in range(action_space)]
        self.q_s_a = [0 for _ in range(action_space)]
        self.prior_probability = 0
        self.num_visits_s_a = [0 for _ in range(action_space)]
        self.num_visits_s = 0

    def uct(self, game, c: float) -> list:

        # Create an array for num_visits_s_a values
        num_visits_s_a_array = np.array(
            [self.num_visits_s_a[a] for a in range(game.getActionSize())]
        )

        # U (s, a) = C(s)P (s, a) N (s)/(1 + N (s, a))
        # need to make sure that this will work as an array
        uct_values = (
            c
            * self.prior_probability
            * (np.sqrt(self.num_visits_s) / (1 + num_visits_s_a_array))
        )

        return uct_values

    def select_action(self, game, c, possible_actions: list) -> int:
        """
        The Upper Confidence bound algorithm for Trees (uct) outputs the
        desirability of visiting a certain node. It is calculated taking into
        account the predicted value of that node as well as the number of
        times that node has been visited in the past to trade off exploration
        and exploitation.

        This function returns the node's child with the highest uct value.

        """
        unexplored_actions = [
            a for a in possible_actions if self.num_visits_s_a[a] == 0
        ]

        # If more than one unvisited child, sample one to visit randomly
        if unexplored_actions:
            return random.choice(unexplored_actions)

        # Otherwise, choose child with the highest uct value
        action_uct = self.uct(game, c=c)

        return np.max(action_uct)


@torch.no_grad()
def apv_mcts(
        game: Game,
        root_state,
        model: torch.nn.Module(),
        num_iterations: int,
        c: float):
    """
    Implementation of the APV-MCTS variant used in the AlphaZero algorithm.

    This algorithm differs from traditional MCTS in that the simulation from
    leaf node to terminal state is replaced by evaluation by a neural network.
    This way, instead of having to play out different scenarios and
    backpropagate received reward, we can just estimate it using a nn and
    backpropagate that estimated value.

    """
    player = 1  # assumption across this system is we're going to start simulations w/ player 1
    for _ in range(num_iterations):
        node = Node(root_state, game.getActionSize())

        # The purpose of this loop is to get us from our current node to a
        # terminal node or a leaf node so that we can either end the game
        # or continue to expand

        path = []
        while node.children and not game.getGameEnded(node.state, player=player):
            possible_actions = game.getValidMoves(node.state)
            if len(possible_actions) > 0:
                action = node.select_action(
                    game=game, possible_actions=possible_actions, c=c
                )
                next_state, player = game.getNextState(
                    board=node.state,
                    player=player,
                    action=action)
                node = Node(next_state, game.getActionSize())

            else:
                logging.info(
                    "Terminal state: no possible actions from %s", (node.state)
                )
                break

            path.append((node, action))

        # expansion phase
        # for our leaf node, expand by adding possible children
        # from that game state to node.children
        if not game.getGameEnded(node.state, player=player):
            # so the model is designed to take in something with dims 8 x 8 x 7
            # this is to include stuff like who the player playing is etc. 
            # currently this doesn't work, need to incorporate that
            policy, _ = model(torch.tensor(node.state, dtype=torch.float32).unsqueeze(0))
            policy = policy.cpu().detach().numpy()
            possible_actions = game.possible_actions(node.state) 
            mask = np.zeros_like(policy, dtype=np.float32)
            mask[possible_actions] = 1

            # we don't want to expand for actions that aren't possible,
            # this allows us to make that distinction
            policy *= mask

            for action, probability in policy:
                if probability > 0:
                    next_state = game.step(action)
                    child = Node(next_state, game.getActionSize())
                    child.prior_probability = probability
                    node.children[action] = child

        _, value = model(torch.tensor(path[-1][0].state, dtype=torch.float32).unsqueeze(0))

        # backpropagation phase
        for node, action in reversed(path):
            node.num_visits_s += 1
            node.num_visits_s_a[action] += 1
            node.total_value_s_a[action] += value
            node.q_s_a[action] = (
                node.total_value_s_a[action] / node.num_visits_s_a[action]
            )
            value = -value  # reverse value we alternates between players
        root = Node(root_state, action_space=game.getActionSize())

        return max(
            root.children, key=lambda child: child.visits
        )  # need to figure this part out
