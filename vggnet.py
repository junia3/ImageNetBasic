import torch
import torch.nn as nn
import torch.nn.functional as F

models = {'VGG11':[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
          'VGG13':[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
          'VGG16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
          'VGG19':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}

class VGGNet(nn.Module):
    def __init__(self, model_cfg='VGG16', num_classes=10):
        super().__init__()

        if model_cfg not in list(models.keys()):
            raise ValueError("Wrong model")

        self.conv_layers = self.build_conv_layers(model_cfg)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
	
        self.initialise()	

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 512*7*7)
        x = self.classifier(x)
        return x

    def build_conv_layers(self, model_cfg):
        layers = []
        in_channels=3 # first input channel
        for name in models[model_cfg]:
            if name == 'M':
                layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))) 
            else:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=name, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
                layers.append(nn.BatchNorm2d(name))
                layers.append(nn.ReLU())
                in_channels = name

        return nn.Sequential(*layers)
        
    def initialise(self):
       for module in self.modules():
           if isinstance(module, nn.Conv2d):
               nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
               if module.bias is not None:
                   nn.init.constant_(module.bias, 0) 

           elif isinstance(module, nn.BatchNorm2d):
               nn.init.constant_(module.weight, 1)
               nn.init.constant_(module.bias, 0)

           elif isinstance(module, nn.Linear):
               nn.init.normal_(module.weight, mean=0, std=0.01)
               nn.init.constant_(module.bias, 0)

