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
from turtle import forward
import torch
import torch.nn as nn
# from torch.autograd import Variable
# from torchvision.datasets import CIFAR10
#########################################################################
# pytorch model
#########################################################################

class Conditioning(nn.Module):
    def __init__(self, feature):
        super(Conditioning, self).__init__()
        # if len(y.shape) < 2:
        #     y = y.unsqueeze(0)
        self.x1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.Sigmoid(),

            nn.Linear(16, 16),
        )
        self.x2 = nn.Linear(16, 16)
        self.feature = feature

    def forward(self, y):
        if len(y.shape) < 2:
            y = y.unsqueeze(0)
        x1 = self.x1(y)
        x2 = self.x2(y)
        conditionZ = x1 * self.feature + x2
        return conditionZ


class Net(nn.Module):
    def __init__(self, inputDim, x, y, is_train, reuse):
        super(Net, self).__init__()
        self.y = y

        # Encoder (E)
        self.encoder = nn.Sequential(
            nn.Flatten(),

            # DenseBlock 128
            nn.Linear(inputDim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # DenseBlock 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # DenseBlock 32
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # DenseBlock 16
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        
        # Decoder (D)
        self.decoder = nn.Sequential(
            # DenseBlock 128
            # nn.Linear(16, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # DenseBlock 128
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # DenseBlock 128
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # DenseBlock 128
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # FIXME: not sure
            nn.Linear(128, inputDim),

            # FIXME: (F, M)
            # Reshape(256, inputDim)
        )

        # # Conditioning (Hr, Hb)
        # self.condition = nn.Sequential(
        #     nn.Linear(16, 16),
        #     nn.Sigmoid(),

        #     nn.Linear(16, 16),
        # )


    def forward(self, x):
        encoded = self.encoder(x)
        conditionZ = Condition(encoded, self.y)
        # condition = self.condition(encoded)
        dbBlock1 = torch.Linear(conditionZ, 128)
        decoded = self.decoder(dbBlock1)
        
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

def Calculate_Loss(x, y, ypred, C, alpha):
    smooth = 1e-6

    # non-match
    ynm_index = torch.where(torch.gt(y, 0))
    ynm = torch.gather(ypred, ynm_index)
    xnm = torch.gather(x, ynm_index)
    ynm = torch.squeeze(ynm, axis=1)
    xnm = torch.squeeze(xnm, axis=1)

    # match
    ym_index = torch.where(torch.lt(y, 0))
    ym = torch.gather(ypred, ym_index)
    xm = torch.gather(x, ym_index)
    ym = torch.squeeze(ym, axis=1)
    xm = torch.squeeze(xm, axis=1)

    loss_nm = torch.mean(torch.abs(ynm - C)) + smooth
    loss_m = torch.mean(torch.abs(ym - xm)) + smooth

    loss = alpha * loss_m + (1 - alpha) * loss_nm

    return loss

def load_model(file_path):
    return torch.load_state_dict(torch.load(file_path))