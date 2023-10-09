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
    def __init__(self, state, parent, action_space) -> None:
        
        self.state = state
        self.terminal = False
        self.children = {}
        self.parent = parent
        
        ''' 
        the following are the quantities required to be stored as per the 
        MCTS implementation in the AlphaGo Zero paper, extended in the 
        AlphaZero paper 
        '''

        self.total_value_s_a = [0 for _ in range(action_space)]
        self.Q_s_a = [0 for _ in range(action_space)]
        self.prior_probabiliy = 0 
        self.num_visits_s_a = [0 for _ in range(action_space)]
        self.num_visits_s = 0 

    def UCT(self, possible_actions: list, c: float) -> list: 

        state = self.state

        policies, value = self.nn.predict(state)

        mask = torch.zeros_like(policies, dtype=torch.float32, device=policies.device)
        mask[possible_actions] = 1
        policies = policies * mask

        policies_np = policies.cpu().detach().numpy()

        # Create an array for num_visits_s_a values
        num_visits_s_a_array = np.array([self.num_visits_s_a[action] for action in range(self.game._get_action_space())])

        # U (s, a) = C(s)P (s, a) N (s)/(1 + N (s, a))
        uct_values = c * self.prior_probabiliy * (np.sqrt(self.num_visits_s[state]) / (1 + num_visits_s_a_array))

        return uct_values

    def select_action(self, state: str, possible_actions: list) -> int:
        '''
        The Upper Confidence bound algorithm for Trees (UCT) outputs the desirability of visiting a certain node. 
        It is calculated taking into account the predicted value of that node as well as the number of times that node 
        has been visited in the past to trade off exploration and exploitation. 

        This function returns the node's child with the highest UCT value. 

        '''
        unexplored_actions = [action for action in possible_actions if self.num_visits_s_a[action]==0]
        
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

    def backpropagate(self, result):
        '''
        Update the current node and propagate back to the root.
        '''
        self.num_visits_s += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(1 - result)

@torch.no_grad()
def apv_mcts(game, root_state, model, num_iterations):
    '''
    Implementation of the APV-MCTS variant used in the AlphaZero algorithm. 

    This algorithm differs from traditional MCTS in that the simulation from 
    leaf node to terminal state is replaced by evaluation by a neural network. 
    This way, instead of having to play out different scenarios and backpropagate 
    received reward, we can just estimate it using a nn and backpropagate that 
    estimated value. 
    
    '''
    root_node = Node(root_state, None)

    for _ in range(num_iterations):
        node = root_node

        # The purpose of this loop is to get us from our current node to a terminal node or a leaf node 
        # so that we can either end the game or continue to expand 

        path = []

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

            path.append((node, action))
            

        #expansion phase
        #for our leaf node, expand by adding possible children from that game state 
        #to node.children
        if not game.is_terminal(node.state):
            
            policy, value = model.predict(node.state)
            policy = policy.cpu().detach().numpy()
            possible_actions = game.possible_actions(node.state)

            mask = np.zeros_like(policy, dtype=np.float32)

            mask[possible_actions] = 1

            #we don't want to expand for actions that aren't possible, this allows us to make that distinction 
            policy *= mask

            for action, probability in policy: 
                if probability > 0: 
                    next_state = game.step(action)
                    child = Node(state=next_state, parent=state)
                    child.prior_probabiliy = probability #according to AGZ paper 
                    node.children[action] = child

        leaf = path[-1][0]
        _, value = model.predict(leaf.state)


        #backpropagation phase 
        for node, action in reversed(path): 
            node.num_visits_s +=1
            node.num_visits_s_a[action]+=1
            node.total_value_s_a[action]+=value
            node.Q_s_a[action] = node.total_value_s_a[action] / node.num_visits_s_a[action]
            value = -value #reverse value as each move alternates between players 
        
        return 