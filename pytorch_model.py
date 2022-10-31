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
        
        nm_label = np.zeros(shape=label.shape)
        
        for i in range(len(label)):
            lb = label[i]
            #nm_indices = [idx for idx in range(len(lb)) if lb[idx] == 0]
            nm_indices = [idx for idx in range(len(lb)) if lb[idx] == 0]
            nm_idx = random.choice(nm_indices)
            nm_label[i][nm_idx] = 1
        
        nm_label = torch.FloatTensor(nm_label).to(device=torch.device('cuda'), non_blocking=True, dtype=torch.float32)
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
    def __init__(self, alpha, C, dim):
        super(CustomLoss, self).__init__()
        
        self.dim = dim
        self.alpha = alpha
        self.const_vector = np.empty(dim)
        self.const_vector.fill(C)
        self.const_vector = torch.Tensor(self.const_vector).to(device=torch.device('cuda'), non_blocking=True, dtype=torch.float32)

    def forward(self, m_output, nm_output, input):
        #print(nm_output[1])
        nm_diff = torch.abs(nm_output - self.const_vector)
        #print(nm_diff[1])
        nm_loss = torch.sum(nm_diff, dim=1)
        #nm_loss = torch.sqrt(nm_dist) 
        #print(nm_loss)
        
        m_diff = torch.abs(m_output - input)
        m_loss = torch.sum(m_diff, dim=1)
        #m_loss = torch.sqrt(m_dist)
        #print(m_loss)

        loss = self.alpha * m_loss + (1 - self.alpha) * nm_loss
        loss = torch.sum(loss)
        return loss