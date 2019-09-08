import torch
import torch.nn as nn
import torchvision

class MirexModel(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.bw2col = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 10, 1, padding=0), nn.ReLU(),
            nn.Conv2d(10, 3, 1, padding=0), nn.ReLU())

        self.mv2 = torchvision.models.mobilenet_v2(pretrained=True)

        self.mv2.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes))

    def forward(self, x):
        x = self.bw2col(x)
        x = self.mv2.features(x)
        x = x.max(dim=-1)[0].max(dim=-1)[0]
        x = self.mv2.classifier(x)
        return x
