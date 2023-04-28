import random


import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch import autograd
from torch.optim.lr_scheduler import ExponentialLR

from torchvision.io import read_image
import pathlib
import cv2
from pathlib import Path
import imageio as iio
import matplotlib.pyplot as plt
import os
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
import sys





class DNN(nn.Module):
    def __init__(self, nin):
        super(DNN, self).__init__()

        self.lin1 = nn.Linear(nin, nin)
        self.lin2 = nn.Linear(nin, 20)
        self.attn1 = nn.Linear(nin, 20)
        self.bn1 = nn.BatchNorm1d(40)

        self.lin3 = nn.Linear(40, 40)
        self.lin4 = nn.Linear(40, 64)
        self.attn2 = nn.Linear(nin, 64)
        self.bn2 = nn.BatchNorm1d(128)

        self.lin5 = nn.Linear(128, 128)
        self.lin6 = nn.Linear(128, 64)
        self.attn3 = nn.Linear(nin, 64)
        self.bn3 = nn.BatchNorm1d(128)

        self.lin7 = nn.Linear(128, 64)
        self.lin8 = nn.Linear(64, 32)
        self.attn4 = nn.Linear(nin, 32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lin9 = nn.Linear(64, 64)
        self.lin10 = nn.Linear(64, 32)
        self.attn5 = nn.Linear(nin, 32)
        self.bn5 = nn.BatchNorm1d(64)

        self.lin11 = nn.Linear(64, 64)
        self.lin12 = nn.Linear(64, 32)
        self.attn6 = nn.Linear(nin, 32)
        self.bn6 = nn.BatchNorm1d(64)

        self.lin13 = nn.Linear(64, 64)
        self.lin14 = nn.Linear(64, 32)
        self.attn7 = nn.Linear(nin, 32)
        self.bn7 = nn.BatchNorm1d(64)

        self.lin15 = nn.Linear(64, 64)
        self.lin16 = nn.Linear(64, 32)
        self.attn8 = nn.Linear(nin, 32)
        self.bn8 = nn.BatchNorm1d(64)

        self.lin17 = nn.Linear(64, 64)
        self.lin18 = nn.Linear(64, 32)
        self.attn9 = nn.Linear(nin, 32)
        self.bn9 = nn.BatchNorm1d(64)

        self.lin19 = nn.Linear(64, 64)
        self.lin20 = nn.Linear(64, 32)
        self.attn10 = nn.Linear(nin, 32)
        self.bn10 = nn.BatchNorm1d(64)

        self.lin21 = nn.Linear(64, 64)
        self.lin22 = nn.Linear(64, 32)
        self.attn11 = nn.Linear(nin, 32)
        self.bn11 = nn.BatchNorm1d(64)

        self.lin23 = nn.Linear(64, 64)
        self.lin24 = nn.Linear(64, 32)
        self.attn12 = nn.Linear(nin, 32)
        self.bn12 = nn.BatchNorm1d(64)

        self.lin25 = nn.Linear(64, 64)
        self.lin26 = nn.Linear(64, 32)
        self.attn13 = nn.Linear(nin, 32)
        self.bn13 = nn.BatchNorm1d(64)

        self.lin27 = nn.Linear(64, 64)
        self.lin28 = nn.Linear(64, 32)
        self.attn14 = nn.Linear(nin, 32)
        self.bn14 = nn.BatchNorm1d(64)

        self.lin29 = nn.Linear(64, 64)
        self.lin30 = nn.Linear(64, 32)
        self.attn15 = nn.Linear(nin, 32)
        self.bn15 = nn.BatchNorm1d(64)

        self.lin31 = nn.Linear(64, 64)
        self.lin32 = nn.Linear(64, 32)
        self.attn16 = nn.Linear(nin, 32)
        self.bn16 = nn.BatchNorm1d(64)

        self.lin33 = nn.Linear(64, 32)
        self.lin34 = nn.Linear(32, 27)
        self.drop = nn.Dropout(p=.05)

    def forward(self, x):
        z = F.relu(self.lin1(x))
        z = F.relu(self.lin2(z))
        z = self.drop(z)
        a = F.relu(self.attn1(x))
        z = torch.cat((z, a), dim=1)
        z = self.bn1(z)

        z = F.relu(self.lin3(z))
        z = F.relu(self.lin4(z))
        z = self.drop(z)
        a = F.relu(self.attn2(x))
        z = torch.cat((z, a), dim=1)
        z = self.bn2(z)

        z = F.relu(self.lin5(z))
        z = F.relu(self.lin6(z))
        z = self.drop(z)
        a = F.relu(self.attn3(x))
        z = torch.cat((z, a), dim=1)
        z = self.bn3(z)

        z = F.relu(self.lin7(z))
        z = F.relu(self.lin8(z))

        z = self.drop(z)
        a = F.relu(self.attn4(x))
        z = torch.cat((z, a), dim=1)
        z = self.bn4(z)

        z = F.relu(self.lin9(z))
        z = F.relu(self.lin10(z))
        z = self.drop(z)
        a = F.relu(self.attn5(x))
        z = torch.cat((z, a), dim=1)
        z = self.bn5(z)

        z = F.relu(self.lin11(z))
        z = F.relu(self.lin12(z))
        z = self.drop(z)
        a = F.relu(self.attn6(x))
        z = torch.cat((z, a), dim=1)
        z = self.bn6(z)

        z = F.relu(self.lin13(z))
        z = F.relu(self.lin14(z))
        z = self.drop(z)
        a = F.relu(self.attn7(x))
        z = torch.cat((z, a), dim=1)
        z = self.bn7(z)


        z = F.relu(self.lin15(z))
        z = F.relu(self.lin16(z))

        z = self.drop(z)
        a = F.relu(self.attn8(x))
        z = torch.cat((z, a), dim=1)
        z = self.bn8(z)


        z = F.relu(self.lin17(z))
        z = F.relu(self.lin18(z))

        z = self.drop(z)
        a = F.relu(self.attn9(x))
        z = torch.cat((z, a), dim=1)
        z = self.bn9(z)


        z = F.relu(self.lin19(z))
        z = F.relu(self.lin20(z))

        z = self.drop(z)
        a = F.relu(self.attn10(x))
        z = torch.cat((z, a), dim=1)
        z = self.bn10(z)


        z = F.relu(self.lin21(z))
        z = F.relu(self.lin22(z))

        z = self.drop(z)
        a = F.relu(self.attn11(x))
        z = torch.cat((z, a), dim=1)
        z = self.bn11(z)

        z = F.relu(self.lin23(z))
        z = F.relu(self.lin24(z))

        z = self.drop(z)
        a = F.relu(self.attn12(x))
        z = torch.cat((z, a), dim=1)
        z = self.bn12(z)

        z = F.relu(self.lin25(z))
        z = F.relu(self.lin26(z))

        z = self.drop(z)
        a = F.relu(self.attn13(x))
        z = torch.cat((z, a), dim=1)
        z = self.bn13(z)

        z = F.relu(self.lin27(z))
        z = F.relu(self.lin28(z))

        z = self.drop(z)
        a = F.relu(self.attn14(x))
        z = torch.cat((z, a), dim=1)
        z = self.bn14(z)

        z = F.relu(self.lin29(z))
        z = F.relu(self.lin30(z))

        z = self.drop(z)
        a = F.relu(self.attn15(x))
        z = torch.cat((z, a), dim=1)
        z = self.bn15(z)

        z = F.relu(self.lin31(z))
        z = F.relu(self.lin32(z))

        z = self.drop(z)
        a = F.relu(self.attn16(x))
        z = torch.cat((z, a), dim=1)
        z = self.bn16(z)

        z = F.relu(self.lin33(z))
        z = self.drop(z)
        z = torch.sigmoid(self.lin34(z))
        return z

xtrain = pd.read_csv('trainset.csv')
ytrain = np.array(xtrain.iloc[:,-1])
xtrain = np.array(xtrain.iloc[:,:-1])
xtest = pd.read_csv('testset.csv')
ytest = np.array(xtest.iloc[:,-1])
xtest = np.array(xtest.iloc[:,:-1])


def binary(x, bits):
    mask = 2**torch.arange(bits-1,-1,-1)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def train(model, optimizer, data, Y, batchsize, offset, device):
    model.train()


    nsamples = np.shape(data)[0]
    cE = 0
    for n in range(0, nsamples, batchsize):

        x = model(torch.tensor(data[n:(n + batchsize)], dtype=torch.float32).to(device))
        y = binary(torch.tensor(Y[n:(n + batchsize)], dtype=torch.int), 27).to(device)
        # y = binary(torch.zeros((1,min(nsamples-n,batchsize)), dtype=torch.int), 27)
        unweightedCEL = torch.abs(y - x)
        weightedCEL = unweightedCEL*((2-offset)**torch.tensor([27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,4,3,2,1,0]))#[1073741824,  536870912,  268435456,  134217728,   67108864,   33554432,262144,     131072,      65536,      32768,      16384,       8192,4096,       2048,       1024,        512,        256,        128,64,         16,          8,          4,          2,          1]

        loss = torch.sum(weightedCEL, dim=1).mean()
        loss.backward()
        # print(model.lin10.weight.grad)
        optimizer.step()
        optimizer.zero_grad()
        nx = (torch.round(x)*(2**torch.tensor([27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,4,3,2,1,0]))).sum(dim=1)
        E = torch.abs(torch.tensor(Y[n:(n + batchsize)], dtype=torch.int)-nx).mean()
        cE = cE+E

        print(nx[0], Y[n])
        print("True Loss (weighted binary):", loss, "Mean Error:", E)
    return model, torch.log(cE/(nsamples/batchsize)+1)
def test(model, data, Y, batchsize, device):
    model.eval()
    nsamples = np.shape(data)[0]
    cE = 0
    for n in range(0, nsamples, batchsize):
        x = model(torch.tensor(data[n:(n + batchsize)], dtype=torch.float32).to(device))
        y = binary(torch.tensor(Y[n:(n + batchsize)], dtype=torch.int), 27).to(device)
        unweightedCEL = torch.abs(y - x)
        weightedCEL = unweightedCEL*(2**torch.tensor([27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,4,3,2,1,0]))#[1073741824,  536870912,  268435456,  134217728,   67108864,   33554432,262144,     131072,      65536,      32768,      16384,       8192,4096,       2048,       1024,        512,        256,        128,64,         16,          8,          4,          2,          1]
        loss = torch.sum(weightedCEL, dim=1).mean()
        x = torch.round(x)*torch.tensor(2**torch.tensor([27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,4,3,2,1,0]))
        x = x.sum(dim=1)
        error = torch.abs(torch.tensor(Y[n:(n + batchsize)], dtype=torch.int) - x).mean()
        cE = cE+error
        print("Sample Guess:", x[0], "True Value:",Y[n])
        print("testLoss", loss, error)
    return torch.log(cE/(nsamples/batchsize)+1)
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    load = True
    if load==False:
        model = DNN(80)
    else:
        model = torch.load('bvmodel.pt')
    model.to(device)
    epochs = 100000
    batchsize = 100
    optimizer = optim.Adam(model.parameters(), lr=.0001)
    scheduler = ExponentialLR(optimizer, gamma=.999)
    TE = []
    TrE = []
    minTE = 1000000
    for super in range(10):
        # off = random.uniform(0, 1)  # (epochs - epoch) / epochs
        for epoch in range(epochs):
            # off = .5*(epochs - epoch) / epochs
            off = 1
            model, trainerr = train(model, optimizer, xtrain, ytrain, batchsize, offset=off, device=device)
            TrE.append(trainerr.item())
            testerr = test(model, xtest, ytest, batchsize, device)
            TE.append(testerr.item())
            scheduler.step()
            plt.clf()
            plt.plot(TrE, label="Train")
            plt.plot(TE, label="Test")
            plt.legend()
            plt.ylabel("Log Error", fontsize=18)
            plt.xlabel("Epoch", fontsize=18)
            plt.show(block=False)
            plt.pause(.01)
            if testerr<minTE:
                torch.save(model, 'bvmodel.pt')

    torch.save(model, 'finalmodel.pt')


