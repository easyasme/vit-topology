import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FCNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FCNet, self).__init__()

        self.fc1 = nn.Linear(9, 3)
        self.fc2 = nn.Linear(3, 4)
        self.fc3 = nn.Linear(4, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)   
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return x

    def forward_features(self, x):
        x = x.view(x.size(0), -1)
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        
        return [x1, x2]
    
    def forward_param_features(self, x):
        return self.forward_features(x)