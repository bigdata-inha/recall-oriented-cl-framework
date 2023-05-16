import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from copy import deepcopy

## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class Lightweight_AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(Lightweight_AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(16, track_running_stats=False)
        # s = compute_conv_output_size(32, 4)
        # s = s // 2
        self.conv2 = nn.Conv2d(16, 32, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(32, track_running_stats=False)
        # s = compute_conv_output_size(s, 3)
        # s = s // 2
        self.conv3 = nn.Conv2d(32, 64, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(64, track_running_stats=False)
        # s = compute_conv_output_size(s, 2)
        # s = s // 2
        self.smid = 2
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * self.smid * self.smid, 512, bias=False)
        self.bn4 = nn.BatchNorm1d(512, track_running_stats=False)
        self.fc2 = nn.Linear(512, 512, bias=False)
        self.bn5 = nn.BatchNorm1d(512, track_running_stats=False)

        self.fc = torch.nn.Linear(512, num_classes, bias=False)


    def forward(self, x):
        bsz = deepcopy(x.size(0))
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x=x.view(bsz,-1)

        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))
        y = self.fc(x)

        return y