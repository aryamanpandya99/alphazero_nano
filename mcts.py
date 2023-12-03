"""
Author: Aryaman Pandya
File contents: Implementation of Monte Carlo Tree Search as well and the
components needed to support it
"""

import logging
import random
from collections import deque
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
        """
        ENTER Docstring
        """
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

    def select_action(self, game, c, valid_moves: np.ndarray) -> int:
        """
        The Upper Confidence bound algorithm for Trees (uct) outputs the
        desirability of visiting a certain node. It is calculated taking into
        account the predicted value of that node as well as the number of
        times that node has been visited in the past to trade off exploration
        and exploitation.

        This function returns the node's child with the highest uct value.

        """
        possible_actions = np.where(valid_moves == 1)[0]
        unexplored_actions = [
            a for a in possible_actions if self.num_visits_s_a[a] == 0
        ]

        # If more than one unvisited child, sample one to visit randomly
        if unexplored_actions:
            return random.choice(unexplored_actions)

        # Otherwise, choose child with the highest uct value
        action_uct = self.uct(game, c=c)

        return np.max(action_uct)


def split_player_boards(board: np.ndarray) -> tuple[np.ndarray, np.ndarray]: 
    """
    ENTER docstring
    """
    player_a = np.maximum(board, 0)
    player_b = board.copy()
    player_b[player_b < 0] = 1

    return player_a, player_b


def update_history_frames(history: np.ndarray, new_frame: np.ndarray, m: int, history_length: int): 
    """
    Updates the history of game boards with a new frame.

    Shifts existing frames in history and adds the new frame at the end.

    Args:
        history (np.ndarray): Game frame history (NxNx(MT+L))
        new_frame (np.ndarray): 2D array representing new game state.
        m (int): Number of channels per frame.
        history_length (int): Number of frames in history.

    Returns:
        None: Updates 'history' array in place.
    """

    # still needs work 
    board_player_1, board_player_2 = split_player_boards(new_frame)
    history[:m*(history_length-1), :,:] = history[m:, :, :]
    new_frames = np.stack([board_player_1, board_player_2], axis=0)
    history[m*(history_length-1):,:,:] = new_frames.reshape(history.shape[0], history.shape[1], m)


def add_player_information(board_tensor: np.ndarray, current_player: int):
    """
    Adds a feature plane indicating the current player.

    Args:
        board_tensor (np.ndarray): The tensor representing the game state.
        current_player (int): The current player (e.g., 0 or 1).

    Returns:
        np.ndarray: Updated board tensor with the player information added.
    """
    # Assuming the last channel is for the current player information
    player_plane = np.full((board_tensor.shape[0], board_tensor.shape[1]), current_player)
    board_tensor[:, :, -1] = player_plane
    return board_tensor

def no_history_model_input(board_arr: np.ndarray, current_player: int) -> np.ndarray: 
    """
    Alternative model input generator. In this function, we take in the game
    board and player and return a stack of boards containing one board per player
    and one board containing current player info. This is an alternative to the method
    in the paper where we store some T length running memory.

    Args: 
        board_arr
        current_player
    Returns: 
        model_input_board
    """
    board_player_1, board_player_2 = split_player_boards(board_arr)
    stacked_board = np.stack((board_player_1, board_player_2), axis = 0)
    if current_player == 1: 
        player_board = np.ones(board_arr.shape)
    else: 
        player_board = np.zeros(board_arr.shape)
    model_input_board = np.concatenate((stacked_board, player_board[np.newaxis,...]), axis = 0)
    return model_input_board

@torch.no_grad()
def apv_mcts(
        game: Game,
        root_state,
        model: torch.nn.Module(),
        num_iterations: int,
        history_length: int,
        c: float, 
        temp = 1):
    """
    Implementation of the APV-MCTS variant used in the AlphaZero algorithm.

    This algorithm differs from traditional MCTS in that the simulation from
    leaf node to terminal state is replaced by evaluation by a neural network.
    This way, instead of having to play out different scenarios and
    backpropagate received reward, we can just estimate it using a nn and
    backpropagate that estimated value.

    """
    n, _ = game.getBoardSize()
    player = 1  # assumption across this system is we're going to start simulations w/ player 1
    root_node = Node(root_state, game.getActionSize())
    input_array = np.zeros((3, 8, 8))
    for _ in range(num_iterations):
        node = root_node
        # The purpose of this loop is to get us from our current node to a
        # terminal node or a leaf node so that we can either end the game
        # or continue to expand
        path = []
        count = 0
        while (len(node.children.keys()) > 0) and not game.getGameEnded(node.state, player=player):
            count+=1
            possible_actions = game.getValidMoves(node.state, player=player)
            if len(possible_actions) > 0:
                action = node.select_action(
                    game=game, valid_moves=possible_actions, c=c
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
                print("terminal state")
                break
            input_array = no_history_model_input(board_arr=node.state, 
                                                 current_player=player)
            path.append((input_array, action))

        # expansion phase
        # for our leaf node, expand by adding possible children
        # from that game state to node.children
        input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)
        if not game.getGameEnded(node.state, player=player):
            # so the model is designed to take in something with dims 8 x 8 x 7
            # this is to include stuff like who the player playing is etc.
            # currently this doesn't work, need to incorporate that
            cannonical_board = game.getCanonicalForm(node.state, player=player)
            policy, _  = model(input_tensor)
            #print(f"value: {val.shape}")
            policy = policy.cpu().detach().numpy().squeeze(0)
            possible_actions = game.getValidMoves(node.state, player=player)
            policy *= possible_actions

            for action_idx, probability in enumerate(policy):
                if probability > 0:
                    next_state = game.getNextState(
                        action=action_idx,
                        board=cannonical_board,
                        player=player
                    )
                    child = Node(next_state, game.getActionSize())
                    child.prior_probability = probability
                    node.children[action_idx] = child

        if len(path) > 0:
            _, value = model(input_tensor)

        # backpropagation phase
        for node, action in reversed(path):
            node.num_visits_s += 1
            node.num_visits_s_a[action] += 1
            node.total_value_s_a[action] += value
            node.q_s_a[action] = (
                node.total_value_s_a[action] / node.num_visits_s_a[action]
            )
            value = -value  # reverse value we alternates between players

    root = path[0][0]
    visits = [x ** 1 / temp for x in root.num_visits_s_a]
    sum_visits = float(sum(visits))
    pi = [x / sum_visits for x in visits]

    return pi
