"""
Author: Aryaman Pandya
File contents: Implementation of Monte Carlo Tree Search as well and the
components needed to support it
"""

import logging
import random

import numpy as np
import torch


class MCTS:
    """
    Class for Monte Carlo Tree Search.
    """

    def __init__(self, game) -> None:
        self.game = game
        self.total_value_s_a = {}
        self.q_s_a = {}
        self.prior_probability = {}
        self.num_visits_s_a = {}
        self.num_visits_s = {}
        self.is_terminal_s = {}

    @torch.no_grad()
    def apv_mcts(
        self,
        canonical_root_state,
        model: torch.nn.Module,
        num_iterations: int,
        device,
        uct_c: float,
        temp=1,
    ):
        """
        Implementation of the APV-MCTS variant used in the AlphaZero algorithm.

        This algorithm differs from traditional MCTS in that the simulation from
        leaf node to terminal state is replaced by evaluation by a neural network.
        This way, instead of having to play out different scenarios and
        backpropagate received reward, we can just estimate it using a nn and
        backpropagate that estimated value.
        """
        input_array = np.zeros((3, 8, 8))
        for _ in range(num_iterations):
            state = canonical_root_state
            player = 1
            state_string = self.game.stringRepresentation(state)
            path = []

            state, state_string = self._find_leaf_or_terminal(
                state, player, path, uct_c, state_string
            )

            input_tensor = (
                torch.tensor(input_array, dtype=torch.float32).to(device).unsqueeze(0)
            )
            game_ended = self.game.getGameEnded(state, player=player)

            if not game_ended:
                _, value = self._expand_leaf(
                    state, player, state_string, model, input_tensor
                )

            else:
                value = game_ended

            value = self._backpropagate(path, value)

        root = path[0][1]
        visits = [x ** (1 / temp) for x in self.num_visits_s_a[root]]
        q_vals = [x ** (1 / temp) for x in self.q_s_a[root]]
        sum_visits = float(sum(visits))
        pi = [x / sum_visits for x in visits]

        return self.uct(root, uct_c)

    def _find_leaf_or_terminal(self, state, player, path, uct_c, state_string):
        """
        This function finds the leaf node or terminal node from the current state
        by traversing the tree.

        Args:
            state: the current state of the game
            player: the current player
            path: the path taken to reach the current state
            uct_c: the uct constant
            state_string: the string representation of the current state

        Returns:
            state: the state of the game at the end of the traversal
            state_string: the string representation of the current state
        """
        while (state_string in self.prior_probability) and not self.game.getGameEnded(
            state, player=player
        ):
            possible_actions = self.game.getValidMoves(state, player=player)
            if len(possible_actions) > 0:
                action = self.select_action(
                    valid_moves=possible_actions, c=uct_c, state_string=state_string
                )
                next_state, player = self.game.getNextState(
                    board=state, player=player, action=action
                )

            else:
                logging.info("Terminal state: no possible actions from %s", (state))
                input_array = self.no_history_model_input(
                    board_arr=state, current_player=player
                )
                path.append((input_array, state_string, action))
                break

            input_array = self.no_history_model_input(
                board_arr=state, current_player=player
            )
            path.append((input_array, state_string, action))
            # print(f'Player: {-player}, state: \n{state}, action: {action}, next_state:\n{next_state}')
            next_state = self.game.getCanonicalForm(next_state, player=player)
            state = next_state
            state_string = self.game.stringRepresentation(state)

        return state, state_string

    def select_action(self, state_string, c, valid_moves: np.ndarray) -> int:
        """
        The Upper Confidence bound algorithm for Trees (uct) outputs the
        desirability of visiting a certain node. It is calculated taking into
        account the predicted value of that node as well as the number of
        times that node has been visited in the past to trade off exploration
        and exploitation.

        This function returns the node's child with the highest uct value.

        """
        possible_actions = np.where(valid_moves == 1)[0]
        if state_string in self.num_visits_s_a:
            unexplored_actions = [
                a for a in possible_actions if self.num_visits_s_a[state_string][a] == 0
            ]
        else:
            unexplored_actions = possible_actions

        if len(unexplored_actions) > 0:
            return random.choice(unexplored_actions)

        action_uct = self.uct(state_string, c=c)

        return np.argmax(action_uct)

    def uct(self, s, c: float) -> list:
        """
        Calculates the Upper Confidence bound for Trees (uct) value for a given state and action.

        Args:
            s (str): The state string.
            c (float): The exploration constant.

        Returns:
            list: The UCT values for all actions.
        """
        # Create an array for num_visits_s_a values
        num_visits_s_a_array = np.array(
            [self.num_visits_s_a[s][a] for a in range(self.game.getActionSize())]
        )
        # U (s, a) = C(s)P (s, a) N (s)/(1 + N (s, a))
        # need to make sure that this will work as an array
        uct_values = (
            c
            * self.prior_probability[s]
            * (np.sqrt(self.num_visits_s[s]) / (1 + num_visits_s_a_array))
        )

        return uct_values

    def _expand_leaf(self, state, player, state_string, model, input_tensor):
        """
        This function expands the leaf node by adding possible children
        from that self.game state to node.children

        Args:
            state: the current state of the game
            player: the current player
            state_string: the string representation of the current state
            model: the neural network model
            input_tensor: the input tensor to the neural network
        """
        model.eval()
        policy, value = model(input_tensor)
        policy = policy.cpu().detach().numpy()

        possible_actions = self.game.getValidMoves(state, player=player)
        policy *= possible_actions

        self.prior_probability[state_string] = policy
        if np.sum(self.prior_probability[state_string]) > 0:
            self.prior_probability[state_string] /= np.sum(
                self.prior_probability[state_string]
            )
        else:
            self.prior_probability[state_string] = (
                self.prior_probability[state_string] + possible_actions
            )
            self.prior_probability[state_string] /= np.sum(
                self.prior_probability[state_string]
            )

        return policy, value

    def _backpropagate(self, path, value):
        """
        This function backpropagates the value received from the neural network
        to the parent nodes.

        Args:
            path: the path taken to reach the current state
            value: the value received from the neural network
        """
        for _, state_string, action in reversed(path):
            if state_string in self.num_visits_s:
                self.num_visits_s[state_string] += 1
            else:
                self.num_visits_s[state_string] = 1
                self.num_visits_s_a[state_string] = [0] * self.game.getActionSize()
                self.total_value_s_a[state_string] = [0] * self.game.getActionSize()
                self.q_s_a[state_string] = [0] * self.game.getActionSize()

            self.num_visits_s_a[state_string][action] += 1
            self.total_value_s_a[state_string][action] += value
            self.q_s_a[state_string][action] = (
                self.total_value_s_a[state_string][action]
                / self.num_visits_s_a[state_string][action]
            )

            value = -value  # reverse value we alternates between players

        return value

    def split_player_boards(self, board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        ENTER docstring
        """
        player_a = np.maximum(board, 0)
        player_b = board.copy()
        player_b = -1 * np.minimum(player_b, 0)

        return player_a, player_b

    def update_history_frames(
        self, history: np.ndarray, new_frame: np.ndarray, m: int, history_length: int
    ):
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
        board_player_1, board_player_2 = self.split_player_boards(new_frame)
        history[: m * (history_length - 1), :, :] = history[m:, :, :]
        new_frames = np.stack([board_player_1, board_player_2], axis=0)
        history[m * (history_length - 1) :, :, :] = new_frames.reshape(
            history.shape[0], history.shape[1], m
        )

    def add_player_information(self, board_tensor: np.ndarray, current_player: int):
        """
        Adds a feature plane indicating the current player.

        Args:
            board_tensor (np.ndarray): The tensor representing the game state.
            current_player (int): The current player (e.g., 0 or 1).

        Returns:
            np.ndarray: Updated board tensor with the player information added.
        """
        # Assuming the last channel is for the current player information
        player_plane = np.full(
            (board_tensor.shape[0], board_tensor.shape[1]), current_player
        )
        board_tensor[:, :, -1] = player_plane
        return board_tensor

    def no_history_model_input(
        self, board_arr: np.ndarray, current_player: int
    ) -> np.ndarray:
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
        board_player_1, board_player_2 = self.split_player_boards(board_arr)
        stacked_board = np.stack((board_player_1, board_player_2), axis=0)

        if current_player == 1:
            player_board = np.ones(board_arr.shape)
        else:
            player_board = np.zeros(board_arr.shape)

        model_input_board = np.concatenate(
            (stacked_board, player_board[np.newaxis, ...]), axis=0
        )

        return model_input_board
