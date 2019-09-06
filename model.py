import torch
import torch.nn as nn

import torchvision
from torchvision import transforms

class Task5Model(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.bw2col = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 10, 1, padding=0), nn.ReLU(),
            nn.Conv2d(10, 3, 1, padding=0), nn.ReLU())

        self.mv2 = torchvision.models.mobilenet_v2(pretrained=True)

        self.final = nn.Sequential(
            nn.Linear(1280, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.bw2col(x)
        x = self.mv2.features(x)
        x = x.max(dim=-1)[0].max(dim=-1)[0]
        x = self.final(x)
        return x