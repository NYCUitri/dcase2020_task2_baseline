"""
 @file   keras_model.py
 @brief  Script for keras model definition
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
#########################################################################
import torch
import torch.nn as nn
import random
# from torch.autograd import Variable
# from torchvision.datasets import CIFAR10
#########################################################################
# pytorch model
#########################################################################
class Net(nn.Module):
    def __init__(self, paramF, paramM, classNum):
        super(Net, self).__init__()

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

        self.condition = FiLMLayer(classNum)

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

    def forward(self, x, label):
        latent = self.encoder(x)
        m_cond_latent = self.condition(label, latent)
        m_output = self.decoder(m_cond_latent)
        
        nm_indices = [idx for idx in range(len(label)) if label[idx] == 0]
        nm_label = np.zeros(shape=label.shape)
        nm_idx = random.choice(nm_indices)
        nm_label[nm_idx] = 1

        nm_cond_latent = self.condition(nm_label, latent)
        nm_output = self.decoder(nm_cond_latent)
        
        return m_output, nm_output

class FiLMLayer(nn.Module):
    def __init__(self, classNum):
        super(FiLMLayer, self).__init__()

        # Conditioning (Hr, Hb)
        # Hadamard Product
        self.condition = nn.Sequential(
            nn.Linear(classNum, 16),
            nn.Sigmoid()
        )

    def forward(self, label, latent): 
        Hb = self.condition(label)
        Hr = self.condition(label)

        cond_latent = latent * Hr + Hb
        return cond_latent

# Reshape Layer
class Reshape(nn.Module): 
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    
    def forward(self, x):
        # return x.view(self.shape)
        return x.view(-1, self.shape[0], self.shape[1])


##############################################################
# Loss Function
import numpy as np
##############################################################
class CustomLoss(nn.Module):
    def __init__(self, alpha, C, n_mels):
        super(CustomLoss, self).__init__()
        
        self.alpha = alpha
        self.const_vector = np.empty(n_mels)
        self.const_vector.fill(C)

    def forward(self, m_output, nm_output, input):
        pass