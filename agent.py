"""
Author: Aryaman Pandya
File contents: Implementation of our AlphaZero Nano agent. This file contains the 
training code and supporting functions required to enable training through self play. 
"""
import sys 
sys.path.append('Othello')

from mcts import apv_mcts
from models import OthelloNN
from othello_game import OthelloGame

class AlphaZeroNano: 
    '''
    Class implementation of AlphaZero agent. 
    '''

    def __init__(self, neural_network, num_simulations, C) -> None:
        
        self.model = neural_network
        self.c_parameter = C 
        self.num_simulations = num_simulations

    def train(self): 
        ## note to self -> use notes to write out trainer 
        pass 
    