import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from typing import Dict, Iterable, Callable
from torchvision import transforms

    
class VTransformer(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, input_size=224):
        super(VTransformer, self).__init__()

        # Load pre-trained model
        if pretrained:
            weights = ViT_B_16_Weights.DEFAULT
            self.preprocess = weights.transforms()
            self.model = vit_b_16(weights=weights)
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.model = vit_b_16(weights=None)

        # Modify classifier head to match classes
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)

        # activations dictionary
        self.activations = {}

        # Register hooks
        self._register_hooks()

    def _get_transform(self):
        return self.preprocess

    # Registers hook on specific layers to capture activations during the forward pass
    def _register_hooks(self):
        # Register forward hooks - encoder blocks
        for idx, block in enumerate(self.model.encoder.layers):
            # register_forward_hook(module, input, output) - pytorch built-in function - capture output
            block.register_forward_hook(self._get_activation(f'encoder_block_{idx}'))
        
        # extract activation at first
        self.model.encoder.ln.register_forward_hook(self._get_activation('encoder_ln'))

    # Creates actual hook function executed during the forward pass
    def _get_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach() # capture activations
        return hook

    # forward pass
    def forward(self, x): # activations from hooked layers captured
        self.activations = {} # reset
        # pass input to model
        output = self.model(x)
        return output

    # retrieves the activations during the pass
    def forward_features(self, x):
        self.forward(x)
        features = [] # collect activations - list of features for each encoder block
        for idx in range(len(self.model.encoder.layers)):
            activation_extracted = self.activations.get(f'encoder_block_{idx}')
            if activation_extracted is not None:
                features.append(activation_extracted)
        return features
