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
            nn.Linear(16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, paramF * paramM)
        )

    def forward(self, x, label, nm_label):
        latent = self.encoder(x)
        m_cond_latent = self.condition(label, latent)
        m_output = self.decoder(m_cond_latent)
        
        nm_cond_latent = self.condition(nm_label, latent)
        nm_output = self.decoder(nm_cond_latent)
        
        return m_output, nm_output

class FiLMLayer(nn.Module):
    def __init__(self, classNum):
        super(FiLMLayer, self).__init__()

        # Conditioning (Hr, Hb)
        # Hadamard Product
        self.condition_r = nn.Sequential(
            nn.Linear(classNum, 16),
            nn.Sigmoid()
        )

        self.condition_b = nn.Sequential(
            nn.Linear(classNum, 16),
            nn.Sigmoid(), 
            nn.Linear(16, 16)
        )

    def forward(self, label, latent): 
        Hr = self.condition_r(label)
        Hb = self.condition_b(label)

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
import torch.nn as nn
##############################################################
class CustomLoss(nn.Module):
    def __init__(self, alpha, C, dim, batch_size):
        super(CustomLoss, self).__init__()
        
        self.alpha = alpha
        self.const_vector = np.empty(shape=(batch_size, dim))
        self.const_vector.fill(C)
        self.const_vector = torch.Tensor(self.const_vector).to(device=torch.device('cuda'), non_blocking=True, dtype=torch.float32)

    def forward(self, m_output, nm_output, input):
        #print(nm_output[1])
        #nm_diff = torch.abs(nm_output - self.const_vector)
        #print(nm_diff[1])
        #nm_loss = torch.sum(nm_diff)
        #nm_loss = torch.sqrt(nm_dist) 
        #print(nm_loss)
        
        #m_diff = torch.abs(m_output - input)

        m_loss = nn.MSELoss()
        nm_loss = nn.MSELoss()
        #m_loss = torch.sum(m_diff)
        #m_loss = torch.sqrt(m_dist)
        #print(m_loss)
        ml = m_loss(m_output, input)

        
        if input.shape[0] < self.const_vector.shape[0]:
            loss = self.alpha * m_loss(m_output, input) + (1 - self.alpha) * nm_loss(nm_output, self.const_vector[:input.shape[0]])
            nml = nm_loss(nm_output, self.const_vector[:input.shape[0]])
        else:
            loss = self.alpha * m_loss(m_output, input) + (1 - self.alpha) * nm_loss(nm_output, self.const_vector)
            nml = nm_loss(nm_output, self.const_vector)
        return loss, ml, nml