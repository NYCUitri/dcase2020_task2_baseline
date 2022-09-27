"""
 @file   keras_model.py
 @brief  Script for keras model definition
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
#########################################################################
from multiprocessing.synchronize import Condition
import torch
import torch.nn as nn
# from torch.autograd import Variable
# from torchvision.datasets import CIFAR10
#########################################################################
# pytorch model
#########################################################################
class Net(nn.Module):
    def __init__(self, inputDim, paramF, paramM):
        super(Net, self).__init__()

        # Encoder (E)
        self.encoder = nn.Sequential(
            # nn.Linear(inputDim, 256),

            nn.Flatten(),

            # DenseBlock
            nn.Linear(inputDim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        
        # Decoder (D)
        self.decoder = nn.Sequential(
            nn.Linear(16, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # FIXME: not sure
            nn.Linear(128, inputDim),

            # FIXME: (F, M)
            # Reshape(256, inputDim)
        )

        # Conditioning (Hr, Hb)
        self.condition = nn.Sequential(
            nn.Linear(16, 16),
            nn.Sigmoid(),

            nn.Linear(16, 16),
        )


    def forward(self, x):
        encoded = self.encoder(x)
        condition = self.condition(encoded)
        decoded = self.decoder(condition)
        
        return decoded
###################################################################

# Reshape Layer
class Reshape(nn.Module): 
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    
    def forward(self, x):
        # return x.view(self.shape)
        return x.view(self.shape[0], -1)

def load_model(file_path):
    return torch.load_state_dict(torch.load(file_path))