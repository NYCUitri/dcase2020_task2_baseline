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

        self.mod_1 = ModulateBlock(classNum, 32, 32)
        self.mod_2 = ModulateBlock(classNum, 32, 32)
        self.mod_3 = ModulateBlock(classNum, 32, 32)
        self.mod_4 = ModulateBlock(classNum, 32, 32)

    def forward(self, latent, label, nm_label):
        match_latent = self.mod_1(label, latent)
        match_latent = self.mod_2(label, match_latent)
        match_latent = self.mod_3(label, match_latent)
        match_latent = self.mod_4(label, match_latent)

        match_latent = self.db_64(match_latent)
        match_latent = self.db_128_1(match_latent)
        match_latent = self.db_128_2(match_latent)
        match_latent = self.db_256(match_latent)

        non_match_latent = self.mod_1(nm_label, latent)
        non_match_latent = self.mod_2(nm_label, non_match_latent)
        non_match_latent = self.mod_3(nm_label, non_match_latent)
        non_match_latent = self.mod_4(nm_label, non_match_latent)

        non_match_latent = self.db_64(non_match_latent)
        non_match_latent = self.db_128_1(non_match_latent)
        non_match_latent = self.db_128_2(non_match_latent)
        non_match_latent = self.db_256(non_match_latent)

        m_output = self.output_layer(match_latent)
        nm_output = self.output_layer(non_match_latent)

        return m_output, nm_output

class FiLMLayer(nn.Module):
    def __init__(self, classNum, size):
        super(FiLMLayer, self).__init__()

        self.size = size
        self.modulate_fn = ConditionBlock(classNum, size)

    def forward(self, label, latent): 
        cond = self.modulate_fn(label)
        #print(cond.shape)
        #print(latent.shape)
        Hr, Hb = cond[:, :self.size], cond[:, self.size:]
        cond_latent = latent * Hr + Hb
        return cond_latent

class ConditionBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConditionBlock, self).__init__()

        self.gamma = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.Sigmoid()
        )

        self.beta = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.Sigmoid()
        )

        self.modulation = nn.Sequential(
            nn.Linear(2*output_size, 2*output_size),
            nn.BatchNorm1d(2*output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        gamma = self.gamma(x)
        beta = self.beta(x)
        modulate_param = torch.cat((gamma, beta), dim=1)   
        output = self.modulation(modulate_param)
        return output

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

class ModulateBlock(nn.Module):
    def __init__(self, classNum, in_size, out_size):
        super(ModulateBlock, self).__init__()

        self.linear = nn.Linear(in_size, out_size)
        self.norm = nn.BatchNorm1d(out_size)
        self.modulation = FiLMLayer(classNum, out_size)
        self.relu = nn.ReLU()

    def forward(self, label, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.modulation(label, x)
        x = self.relu(x)
        return x
##############################################################