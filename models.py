"""
Author: Aryaman Pandya
File contents: Convolutional neural network that takes in board state 
and outputs expected value 
"""

from torch import nn
import numpy as np


class othello_model(nn.Module):
    def __init__(self) -> None:

        super(othello_model, self).__init__()

        # we expect an input of dimensionality 8 x 8 x 7
        # following conventions from the paper:
        # N x N -> 8 x 8. M = 2, T = 3, L =1 (who's playing)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=128, kernel_size=3, padding=1),
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

        self.policy_head = nn.Sequential(nn.Linear(8192, 64), nn.Softmax(dim=1))

        self.value_head_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.value_head_linear = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh()
        )

    def forward(self, state) -> tuple(np.array, int):
        pass
