# ImageNetBasic
Deep learning의 Computer Vision에서 Image classification의 기초가 되는 세 모델을 소개합니다

이미지넷에서 가장 먼저 딥러닝의 가능성을 보여준 AlexNet을 시작으로
보다 깊은 네트워크로 성능을 높인 VGGNet(3*3 convolution layer를 여러 층 사용),
그리고 이보다 더 깊은 네트워크를 위해서 Residual network를 구현한 ResNet를 모두 pytorch로 구현해보았습니다.
Dataset으로는 10개의 클래스를 가지는 STL10을 사용해보았는데, 이후 CIFAR10이나 MNIST등 다양한 데이터셋도 추가해볼 생각입니다.


---
# 실행법

먼저, 필요한 라이브러리 설치 (pytorch, torchvision, torchsummary 등등 다양한 라이브러리가 필요합니다.)
본인 컴퓨터의 cuda 버전에 맞춰 잘 설치해주세요.

Main이 되는 python파일은 run.py입니다.
그리고 available한 모델을 직접 입력하게 되면 알아서 dataset을 설정하고, training을 진행합니다.

---
# AlexNet
```python3
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
```

AlexNet은 본래 GPU 두 개에 각각의 채널을 분리하여 병렬 연산을 하였으나, 이를 직접 구현하지는 않고 한꺼번에 구현하였다.
STL10 데이터셋으로 Training 및 Validation을 하였는데, 결과가 그리 좋지 못하다. 결과가 나오면 첨부할 예정

---
# VGGNet
```python3
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
```
---
# ResNet
```python3
class ShortCut(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion):
        super().__init__()
        if stride != 1 or in_channels != expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*expansion),
            )
        else:
            self.shortcut = nn.Sequential() 

    def forward(self, x):
        return self.shortcut(x)


class ResidualBlock(nn.Module):
    expansion=1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels*ResidualBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels*ResidualBlock.expansion),
        )
        
        self.shortcut = ShortCut(in_channels, out_channels, stride, ResidualBlock.expansion)
        

    def forward(self, x):
        x = self.residual_network(x) + self.shortcut(x)
        x = F.relu(x)
        return x

class BottleNeckResidualBlock(nn.Module):
    expansion=4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels*BottleNeckResidualBlock.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels*BottleNeckResidualBlock.expansion),
        )

        self.shortcut = ShortCut(in_channels, out_channels, stride, BottleNeckResidualBlock.expansion)

    def forward(self, x):
        x = self.shortcut(x) + self.residual_network(x)
        return F.relu(x)

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=10):
        super().__init__()
        self.in_channels=64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self.build_layer(block, 64, num_block[0], 1)
        self.conv3 = self.build_layer(block, 128, num_block[1], 2)
        self.conv4 = self.build_layer(block, 256, num_block[2], 2)
        self.conv5 = self.build_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512*block.expansion, num_classes) 
      
        self.initialise()   
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def initialise(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    n.init.constant(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0) 

    def build_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride]+[1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels*block.expansion
        
        return nn.Sequential(*layers)
```
