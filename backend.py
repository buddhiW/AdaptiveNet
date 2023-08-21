"""
Author: Buddhi Wickramasinghe

Based on the models proposed in:
Alzantot, Moustafa, Ziqi Wang, and Mani B. Srivastava. "Deep residual neural networks for audio spoofing detection."
arXiv preprint arXiv:1907.00501 (2019).
Link: https://github.com/nesl/asvspoof2019
"""

import torch
from torch import nn

class ResNetBlock(nn.Module):
    def __init__(self, in_depth, depth, first=False):
        super(ResNetBlock, self).__init__()
        self.first = first
        self.conv1 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(depth)
        self.lrelu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(depth, depth, kernel_size=3, stride=2, padding=1)
        self.conv11 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=2, padding=1)
        if not self.first:
            self.pre_bn = nn.BatchNorm2d(in_depth)

    def forward(self, x):
        # x is (B x d_in x T)
        prev = x
        prev_mp = self.conv11(x)
        if not self.first:
            # print(self.first)
            out = self.pre_bn(x)
            out = self.lrelu(out)
        else:
            out = x
        out = self.conv1(out)
        # out is (B x depth x T/2)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        # out is (B x depth x T/2)
        out = out + prev_mp
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = ResNetBlock(32, 32, True)
        self.mp = nn.MaxPool2d(2, stride=2, padding=1)
        self.block2 = ResNetBlock(32, 32, False)
        self.block3 = ResNetBlock(32, 32, False)
        self.block4 = ResNetBlock(32, 32, False)
        self.block5 = ResNetBlock(32, 32, False)
        self.block6 = ResNetBlock(32, 32, False)
        self.block7 = ResNetBlock(32, 32, False)
        self.block8 = ResNetBlock(32, 32, False)
        self.block9 = ResNetBlock(32, 32, False)
        # self.block10 = ResNetBlock(32, 32, False)
        self.block11 = ResNetBlock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(192, 128)  # Num_frames=401:448 #Num_frames=301:384 #Num_frames=100:192
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)  # [32,1,90,469]
        out = self.conv1(x)  # [32, 32, 90, 469]
        out = self.block1(out)  # [32, 32, 30, 157]
        # out = self.block2(out)
        # out = self.mp(out)
        out = self.block3(out)  # [32, 32, 10, 53]
        # out = self.block4(out)
        # out = self.mp(out)
        out = self.block5(out)  # [32, 32, 4, 18]
        # out = self.block6(out)
        # out = self.mp(out)
        out = self.block7(out)  # [32, 32, 2, 6]
        # out = self.block8(out)
        # out = self.mp(out)
        out = self.block9(out)  # [32, 32, 1, 2]
        # out = self.block10(out)
        # out = self.mp(out)
        # out = self.block11(out)
        out = self.bn(out)
        out = self.lrelu(out)  # [32, 32, 1, 2]
        out = self.mp(out)
        # print(x.shape)
        # print(out.shape)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out
