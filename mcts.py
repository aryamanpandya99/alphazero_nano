'''
Author: Aryaman Pandya
File contents: Implementation of Monte Carlo Tree Search as well and the 
components needed to support it 
'''

import math
import random
import numpy as np 

class MCTS(object): 


    def __init__(self, model, exploration_factor) -> None:
        
        self.num_visits_s_a = {}
        self.total_value_s_a = {}
        self.mean_value_s_a = {}
        self.policies_s = {}
        self.num_visits_s = {}
        self.nn = model 

        self.exploration_factor = exploration_factor

    def UCT(self, state: str, possible_actions: list, c: float) -> list: 

        policies, value = self.nn(state)

        for idx, policy_value in enumerate(policies):
            if idx not in possible_actions:
                policies[idx] = 0

        self.policies_s[state] = policies

        # U (s, a) = C(s)P (s, a) N (s)/(1 + N (s, a))
        

        return []

    def select_action(self, state: str) -> int:
        '''
        The Upper Confidence bound algorithm for Trees (UCT) outputs the desirability of visiting a certain node. 
        It is calculated taking into account the predicted value of that node as well as the number of times that node 
        has been visited in the past to trade off exploration and exploitation. 

        This function returns the node's child with the highest UCT value. 

        '''

        possible_actions = self.game.possible_actions(state)
        unexplored_actions = [action for action in possible_actions if self.num_visits_s_a[(state, action)]==0]
        
        #If there's more than one unvisited child, sample one to visit randomly. 
        if unexplored_actions:
            return random.choice(unexplored_actions)
        
        # Otherwise, choose child with the highest UCT value
        action_UCT = self.UCT(state, possible_actions)
        return max(self.children, key=lambda child: child.wins / child.visits + c * math.sqrt(2 * math.log(self.visits) / child.visits))
    
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
        root = Node(root_state)

        for _ in range(num_iterations):
            node = root
            state = root_state
        
            # The purpose of this loop is to get us from our current node to a terminal node or a leaf node 
            # so that we can either end the game or continue to expand 
            while node.children and not state.is_terminal():
                node = node.select_child()
                state = node.state

            # Expand if the reason for the above exit was not termination 
            if not state.is_terminal():
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