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
    def search(self, root_state, num_iterations):
        '''
        
        
        What's left to do from a skeletal implementation perspective: 
        1. Update probabilities_s when visiting a new leaf state 
        2. figure out what to do with predicted value..? 
        3. Initialize pairs to 0 
    
        
        '''

        for _ in range(num_iterations):
            state = root_state

            # The purpose of this loop is to get us from our current node to a terminal node or a leaf node 
            # so that we can either end the game or continue to expand 
            while not self.game.is_terminal(state):

                if self.num_visits_s.get(state, 0) == 0:
                    break
                
                possible_actions = self.game.possible_actions(state)

                if len(possible_actions) > 0: 
                    action = self.select_action(state=state, possible_actions=possible_actions)
                    next_state = self.game.step(action)
                    state = next_state 

                else: 
                    logging.info(f"Terminal state detected: no possible actions from state {state}")
                    break
                

            #simulation phase
            traversal_history = []
            while not self.game.is_terminal(state):
                
                possible_actions = self.game.possible_actions(state)

                action = random.choice(possible_actions())
                next_state = self.game.step(action)
                
                state = next_state
            
            reward = self.game.reward(state)
            self.backpropagate(reward, traversed_states=traversal_history)

                



        return 