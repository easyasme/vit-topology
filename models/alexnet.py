import torch 
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, input_size=32):
        super(AlexNet, self).__init__()
        
        self.feat_size = 256 if input_size==32 else (256 * 2 * 2) if input_size==64 else -1

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
         )   
        '''
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.feat_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        '''
        self.classifier = nn.Linear(self.feat_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

    def forward_features(self, x):
        layers = [2, 5, 7, 9, 12]
        feats = [self.features[:l+1](x) for l in layers] + [self.forward(x)]
        
        return feats
