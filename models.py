"""
Author: Aryaman Pandya
File contents: Convolutional neural network that takes in board state 
and outputs expected value 
"""

from torch import nn
import numpy as np


class OthelloNN(nn.Module):
    '''
    Convolutional neural network used in the AlphaZero implementation scaled 
    for the dimensions of the othello game. 
    '''
    def __init__(self) -> None:
        '''
        Initialization of the neural network graph. 
        Contains a common body that includes 4 sequential Conv2D operations 
        followed by batch normalization and ReLU. 
        Contains two separate heads for policy and value estimation as specified 
        in the original DeepMind paper supplemental materials. 
        '''

        super().__init__()
        #we expect an input of dimensionality 8 x 8 x 7 following conventions from the paper:
        #N x N -> 8 x 8. M = 2, T = 3, L =1 (who's playing)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )


        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 65),  # Flatten the conv output and connect to a Linear layer
            nn.Softmax(dim=1)
        )

        self.value_head_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.value_head_linear = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 1), nn.ReLU(), nn.Tanh()
        )

    def forward(self, state) -> tuple[np.array, int]:
        '''
        Forward pass for the nn graph

        Args:
            param1: self
            param2: state- game state at time of evaluation 

        Returns:
            pi (torch.tensor): policy pi[a|s]
            val (float32): scalar value estimate from input state 
        '''
        s = self.conv1(state)
        s = self.conv2(s)
        s = self.conv3(s)
        s = self.conv4(s)

        pi = self.policy_head(s)
        s = self.value_head_conv(s)
        val = self.value_head_linear(s).squeeze()

        return pi, val
