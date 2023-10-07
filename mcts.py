'''
Author: Aryaman Pandya
File contents: Implementation of Monte Carlo Tree Search as well and the 
components needed to support it 
'''

import math
import random
import numpy as np 
import torch 
import logging 


class Node(object): 
    def __init__(self, state, parent) -> None:
        
        self.state = state
        self.terminal = False
        self.children = {}
        self.parent = parent
        
        ''' 
        the following are the quantities required to be stored as per the 
        MCTS implementation in the AlphaGo Zero paper, extended in the 
        AlphaZero paper 

        we use dictionaries to use action as a key to easily access values 
        for evaluation. We could alternatively initialize lists with length 
        equal to number of actions, but this way we save some memory 
        '''
        self.total_value_s_a = {}
        self.mean_value_s_a = {} 
        self.prior_probabiliy = 0 
        self.num_visits_s_a = {}
        self.num_visits_s = 0 

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

    def simulate(self, root): 
        '''
        Once we reach an unvisited leaf, we simulate a traversal from it to a terminal state 
        '''

    def backpropagate(self, result, traversed_states):
        '''
        Update the current node and propagate back to the root.
        '''
        for state, action in reversed(traversed_states):
            
            self.num_visits_s[state] += 1
            self.num_visits_s_a[(state, action)] += 1
            self.total_value_s_a[(state, action)] += result
            self.mean_value_s_a[(state, action)] = self.total_value_s_a[(state, action)] / self.num_visits_s_a[(state, action)]

@torch.no_grad()
def apv_mcts(game, root_state, model, num_iterations):
    '''

    
    '''
    root_node = Node(root_state, None)

    for _ in range(num_iterations):
        node = root_node

        # The purpose of this loop is to get us from our current node to a terminal node or a leaf node 
        # so that we can either end the game or continue to expand 

        while node.children and not game.is_terminal(state): 
            
            possible_actions = game.possible_actions(state)

            if len(possible_actions) > 0: 
                action = node.select_action(state=state, possible_actions=possible_actions)
                next_state = game.step(action)
                node = Node(state=next_state, parent=state)
                state = next_state 

            else: 
                logging.info(f"Terminal state detected: no possible actions from state {state}")
                break
            

        #expansion phase
        if not game.is_terminal(node.state):
            
            policy, value = model.predict(node.state)
            policy = policy.cpu().detach().numpy()
            possible_actions = game.possible_actions(node.state)

            mask = np.zeros_like(policy, dtype=np.float32)

            mask[possible_actions] = 1

            policy *= mask

            for action, probability in policy: 
                if probability > 0: 
                    next_state = game.step(action)
                    child = Node(state=next_state, parent=state)
                    child.prior_probabiliy = probability
                    node.children[action] = child

        
        node.backpropagate(value)
        
        return 