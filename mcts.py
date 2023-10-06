'''
Author: Aryaman Pandya
File contents: Implementation of Monte Carlo Tree Search as well and the 
components needed to support it 
'''

import math
import random

class Node:

    '''
        Declaration of a tree node. Each node has some internal state, a list of children and a parent. 
        We also keep track of the number of visits made to each node and number of wins after having 
        visited said node. 
    
    '''
    
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

    
    def select_child(self, c=1.4):
        '''
        The Upper Confidence bound algorithm for Trees (UCT) outputs the desirability of visiting a certain node. 
        It is calculated taking into account the predicted value of that node as well as the number of times that node 
        has been visited in the past to trade off exploration and exploitation. 

        This function returns the node's child with the highest UCT value. 

        '''

        if not self.children:
            return None
        
        unvisited_children = [child for child in self.children if child.visits == 0]
        
        #If there's more than one unvisited child, sample one to visit randomly. 
        if unvisited_children:
            return random.choice(unvisited_children)
        
        # Otherwise, choose child with the highest UCT value
        return max(self.children, key=lambda child: child.wins / child.visits + c * math.sqrt(2 * math.log(self.visits) / child.visits))

    
    def expand(self):
        '''
        This function takes all possible actions, and then adds the output state (next_state) of the given state-action pair 
        to the list of children for our current state node. 
        '''
        for action in self.state.possible_actions():
            next_state = self.state.step(action)
            self.children.append(Node(next_state, parent=self))

    def backpropagate(self, result):
        '''
        Update the current node and propagate back to the root.
        '''
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(1 - result)


class MCTS(object): 


    def __init__(self, model) -> None:
        
        self.num_visits = {}
        self.total_value = {}
        self.mean_value = {}
        self.probabilities_prior = {}
        self.nn = model 



def mcts(root_state, num_iterations):
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