'''
Author: Aryaman Pandya
File contents: Implementation of Monte Carlo Tree Search as well and the 
components needed to support it 
'''

import math
import random
import numpy as np 
import torch 

class MCTS(object): 


    def __init__(self, model, exploration_factor) -> None:
        
        
        self.total_value_s_a = {}
        self.mean_value_s_a = {}
        
        self.policies_s = {}
        
        self.num_visits_s_a = {}
        self.num_visits_s = {}

        self.terminal_states = {}

        self.nn = model 

        self.exploration_factor = exploration_factor

    def UCT(self, state: str, possible_actions: list, c: float) -> list: 

        policies, value = self.nn(state)

        mask = torch.zeros_like(policies, dtype=torch.float32, device=policies.device)
        mask[possible_actions] = 1
        policies = policies * mask

        policies_np = policies.cpu().detach().numpy()

        self.policies_s[state] = policies_np

        # Create an array for num_visits_s_a values
        num_visits_s_a_array = np.array([self.num_visits_s_a[(state, action)] for action in range(self.game._get_action_space())])

        # U (s, a) = C(s)P (s, a) N (s)/(1 + N (s, a))
        uct_values = c * self.policies_s[state] * (np.sqrt(self.num_visits_s[state]) / (1 + num_visits_s_a_array))

        return uct_values

    def select_action(self, state: str, possible_actions: list) -> int:
        '''
        The Upper Confidence bound algorithm for Trees (UCT) outputs the desirability of visiting a certain node. 
        It is calculated taking into account the predicted value of that node as well as the number of times that node 
        has been visited in the past to trade off exploration and exploitation. 

        This function returns the node's child with the highest UCT value. 

        '''
        unexplored_actions = [action for action in possible_actions if self.num_visits_s_a[(state, action)]==0]
        
        #If there's more than one unvisited child, sample one to visit randomly. 
        if unexplored_actions:
            return random.choice(unexplored_actions)
        
        # Otherwise, choose child with the highest UCT value
        action_UCT = self.UCT(state, possible_actions)
        
        return np.max(action_UCT)
    
    def expand(self) -> None:
        '''
        This function takes all possible actions, and then adds the output state (next_state) of the given state-action pair 
        to the list of children for our current state node. 
        '''
        for action in self.state.possible_actions():
            next_state = self.state.step(action)
            self.children.append(Node(next_state, parent=self))

    def simulate(self, root): 
        '''
        Once we reach an unvisited leaf, we simulate a traversal from it to a terminal state 
        '''

    def backpropagate(self, result):
        '''
        Update the current node and propagate back to the root.
        '''
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(1 - result)


    def search(self, root_state, num_iterations):
        #set root node to a node with the specified root state. Presumably, this will be the starting game board. 

        for _ in range(num_iterations):
            state = root_state

            # The purpose of this loop is to get us from our current node to a terminal node or a leaf node 
            # so that we can either end the game or continue to expand 
            while not self.game.is_terminal(state):
                
                possible_actions = self.game.possible_actions(state)

                if len(possible_actions) > 0: 
                    action = self.select_action(state=state, possible_actions=possible_actions)
                    next_state = self.game.step(action)
                    state = next_state 
                
                else: 
                    print(f"Terminal state detected: no possible actions from state {state}")
                    break
                

            # Expand if the reason for the above exit was not termination 
            if not self.game.is_terminal(state):
                node.expand()
                if node.children:
                    node = node.children[0]
                    state = node.state

            #simulation phase
            while not state.is_terminal():
                action = random.choice(state.possible_actions())
                state = state.step(action)


            #backprop
            winner = state.get_winner()
            node.backpropagate(winner)

        return max(root.children, key=lambda child: child.visits).state.board