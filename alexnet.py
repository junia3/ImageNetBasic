import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super().__init__()
        # input size = batch * 3 * 227 * 227
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4), # batch * 96 * 55 * 55
            nn.ReLU(inplace=False),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2), # batch * 96 * 27 * 27
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), # batch * 256 * 27 * 27
            nn.ReLU(inplace=False),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2), # batch * 256 * 13 * 13
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), # batch * 384 * 13 * 13
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), # batch * 384 * 13 * 13
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1), # batch * 256 * 13 * 13
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2), # batch * 256 * 6 * 6
            nn.AdaptiveAvgPool2d((6, 6)) # adaptive average pooling
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

        self.initialise()

    def initialise(self):
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.layers[4].bias, 1)
        nn.init.constant_(self.layers[10].bias, 1)
        nn.init.constant_(self.layers[12].bias, 1)

        for fc_layer in self.classifier:
            if isinstance(fc_layer, nn.Linear):
                nn.init.normal_(fc_layer.weight, mean=0, std=0.01)
                nn.init.constant_(fc_layer.bias, 0)


    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 256*6*6)
        return self.classifier(x)
