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
class Encoder(nn.Module):
    def __init__(self, paramF, paramM, classNum):
        super(Encoder, self).__init__()

        self.db_256 = DenseBlock(paramF * paramM, 256)
        self.db_128 = DenseBlock(256, 128)
        self.db_64 = DenseBlock(128, 64)
        self.db_32 = DenseBlock(64, 32)

        self.classifier = nn.Sequential(
            nn.Linear(32, classNum), 
            nn.Softmax(dim=1)
        ) 
    
    def forward(self, x):
        x = self.db_256(x)
        x = self.db_128(x)
        x = self.db_64(x)
        latent = self.db_32(x)

        cls_output = self.classifier(latent)
        return latent, cls_output

class Decoder(nn.Module):
    def __init__(self, paramF, paramM, classNum):
        super(Decoder, self).__init__()

        self.db_64 = DenseBlock(32, 64)
        self.db_128_1 = DenseBlock(64, 128)
        self.db_128_2 = DenseBlock(128, 128)
        self.db_256 = DenseBlock(128, 256)
        self.output_layer = nn.Linear(256, paramM * paramF)

        self.film_32 = FiLMLayer(classNum, 32)
        self.film_64 = FiLMLayer(classNum, 64)
        self.film_128_1 = FiLMLayer(classNum, 128)
        self.film_128_2 = FiLMLayer(classNum, 128)

    def forward(self, latent, label, nm_label):
        match_latent = self.film_32(label, latent)
        match_latent = self.db_64(match_latent)

        match_latent = self.film_64(label, match_latent)
        match_latent = self.db_128_1(match_latent)

        match_latent = self.film_128_1(label, match_latent)
        match_latent = self.db_128_2(match_latent)

        match_latent = self.film_128_2(label, match_latent)
        match_latent = self.db_256(match_latent)

        non_match_latent = self.film_32(nm_label, latent)
        non_match_latent = self.db_64(non_match_latent)

        non_match_latent = self.film_64(nm_label, non_match_latent)
        non_match_latent = self.db_128_1(non_match_latent)

        non_match_latent = self.film_128_1(nm_label, non_match_latent)
        non_match_latent = self.db_128_2(non_match_latent)

        non_match_latent = self.film_128_2(nm_label, non_match_latent)
        non_match_latent = self.db_256(non_match_latent)


        m_output = self.output_layer(match_latent)
        nm_output = self.outpu_layer(non_match_latent)

        return m_output, nm_output

    """ def predict(self, x, label):
        cond_latent = self.condition(label, latent)
        output = self.decoder(cond_latent)

        return output """

class FiLMLayer(nn.Module):
    def __init__(self, classNum, size):
        super(FiLMLayer, self).__init__()

        self.size = size
        # Conditioning (Hr, Hb)
        # Hadamard Product
        self.film = nn.Sequential(
            nn.Linear(classNum, size),
            nn.BatchNorm1d(size),
            nn.ReLU(), 

            nn.Linear(size, 2*size),
            nn.BatchNorm1d(2*size),
            nn.ReLU()
        )

    def forward(self, label, latent): 
        cond = self.film(label)
        Hr, Hb = cond[:self.size], cond[self.size:]

        cond_latent = latent * Hr + Hb
        return cond_latent
class DenseBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(DenseBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.BatchNorm1d(out_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.block(x)
##############################################################
# Loss Function
import numpy as np
import torch.nn as nn
##############################################################
class CustomLoss(nn.Module):
    def __init__(self, C, dim, batch_size):
        super(CustomLoss, self).__init__()
        
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

        loss_fn = nn.L1Loss()
        #m_loss = torch.sum(m_diff)
        #m_loss = torch.sqrt(m_dist)
        #print(m_loss)
        m_loss = loss_fn(m_output, input)

        if input.shape[0] < self.const_vector.shape[0]:
            nm_loss = loss_fn(nm_output, self.const_vector[:input.shape[0]])
        else:
            nm_loss = loss_fn(nm_output, self.const_vector)

        return m_loss, nm_loss