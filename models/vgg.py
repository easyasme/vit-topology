import torch.nn as nn

cfg = {
    'VGG9':  [64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }


'''
if self.name == 'VGG11': layers = [3, 7, 10, 14, 17, 21, 24, 27, 29]
if self.name == 'VGG16': layers = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 44]
if self.name == 'VGG16': layers = [19, 23, 26, 29, 33, 36, 39, 44]
'''    


class VGG(nn.Module):
    def __init__(self, vgg_name, img_size=28, num_classes=10):
        super(VGG, self).__init__()
        
        self.img_size = img_size
        self.feats_size = (img_size**2) // 2 if img_size == 64 else 512
        self.name = vgg_name
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(self.feats_size, num_classes)
        
    def forward(self, x, mask=None):
        layers = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
        if mask is not None:
            x = self.features[:layers[0]+1](x)
            for layer_begin, layer_end, m in zip(layers[:-1],layers[1:], mask[1:]):
                x = self.features[layer_begin+1:layer_end+1](x)
                x = x*m                
            x = self.features[layer_end+1:](x)
        else:
            x = self.features(x)

        # print(f'Size of x:', x.size())
        out = x.view(x.size(0), -1)
        # print(f'Size of out:', out.size())
        out = self.classifier(out)
        
        return out

    def forward_features(self, x):
        ''' Forward selected features '''
        if self.name == 'VGG16': layers = [13, 19, 26, 33, 39]
        
        return [self.features[:l+1](x) for l in layers] + [self.forward(x)]
        
    def forward_param_features(self, x):
        ''' Forward features from all convolutional and linear layers '''
        if self.name == 'VGG16': layers = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
        
        return [self.features[:l+1](x) for l in layers] + [self.forward(x)]

    def forward_all_features(self, x):
        return [self.features[:l+1](x) for l in range(45)] + [self.features]
        
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3 if self.img_size != 28 else 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        
        return nn.Sequential(*layers)
