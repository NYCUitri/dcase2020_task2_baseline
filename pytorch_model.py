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
class Encoder(nn.moduel):
    def __init__(self, paramF, paramM):
        super(Encoder, self).__init__()

        # Encoder (E)
        self.encoder = nn.Sequential(
            #nn.Flatten(),

            # DenseBlock
            nn.Linear(paramF * paramM, 128),
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
            nn.ReLU()
        )
    
    def foward(self, x):
        return self.encoder(x)

class Condition(nn.Module):
    def __init__(self, classNum):
        super(Condition, self).__init__()

        # Conditioning (Hr, Hb)
        # Hadamard Product
        self.condition = nn.Sequential(
            nn.Linear(classNum, 16),
            nn.Sigmoid(),
            nn.Linear(16, 16),
        )

    def forward(self, x): 
        labeled_latent = self.condition(x)
        return labeled_latent

class Decoder(nn.Module):
    def __init__(self, paramF, paramM):
        super(Decoder, self).__init__()

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

            nn.Linear(128, paramF * paramM)
        )

        #self.reshape = Reshape((paramF, paramM))

    def foward(self, x):
        decoded = self.decoder(x)
        #output = self.reshape(decoded)
        return decoded

# Reshape Layer
class Reshape(nn.Module): 
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    
    def forward(self, x):
        # return x.view(self.shape)
        return x.view(-1, self.shape[0], self.shape[1])
