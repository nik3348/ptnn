import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(128, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(128, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_rate),
            nn.Flatten(),
            nn.BatchNorm1d(256)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
