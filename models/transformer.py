import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from typing import Dict, Iterable, Callable


# class FeatureExtractor(nn.Module):
#     def __init__(self, model: nn.Module, layers: Iterable[str]):
#         super(FeatureExtractor, self).__init__()

#         self.model = model
#         self.layers = layers
#         self._features = {layer: torch.empty(0) for layer in layers}

#         for layer_id in layers:
#             layer = dict([*self.model.named_modules()])[layer_id]
#             layer.register_forward_hook(self.save_outputs_hook(layer_id))

#     def save_outputs_hook(self, layer_id: str) -> Callable:
#         def fn(_, __, output):
#             self._features[layer_id] = output
#         return fn

#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         _ = self.model(x)
#         return self._features
    
class VTransformer(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, input_size=224):
        super(VTransformer, self).__init__()

        # Load pre-trained model
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            print(weights)
            self.model = vit_b_16(weights=weights)
        else:
            self.model = vit_b_16(weights=None)

        # Modify classifier head to match classes
        # self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)

        # activations dictionary
        self.activations = {}

        # Register hooks
        self._register_hooks()

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

    # Forward hooks remain properly registered and active when switching to two modes
    def train(self, mode=True):
        self.model.train(mode)
        return super(VTransformer, self).train(mode)
    
    def eval(self):
        self.model.eval()
        return super(VTransformer, self).eval()

# def test_vtransformer():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     weights = ViT_B_16_Weights.IMAGENET1K_V1
#     preprocess = weights.transforms()

#     model = VTransformer(num_classes=10, pretrained=True, img_size=224)

#     model.to(device)
#     model.eval()

#     example_input = torch.randn(2, 3, 224, 224)
#     example_input = preprocess(example_input).to(device)
    
#     children = model.named_parameters()
#     layers = [name for name, _ in children if 'encoder_layer.\*.mlp' in name]
#     for child in layers:
#         print(f'Child: {child.__str__()}')

#     extractor = FeatureExtractor(model, layers)
#     activations = extractor(example_input)
#     print(f'Activations: {activations.values()}')
#     tensor_lst = [tensor for tensor in activations.values()]
#     tensor_lst = torch.cat(tensor_lst, dim=1)
#     for key in activations.keys():
#         print(f'Layer: {key}, Activations size: {activations[key]}')

#     print(f'Activations size: {activations.size()}')

# if __name__ == "__main__":
#     test_vtransformer()

# class VTransformer(nn.Module):
#     def __init__(self, num_classes=10, pretrained=True, input_size=224): # how many num_classes?
#         super(VTransformer, self).__init__()

#         #Load pre-trained model
#         if pretrained:
#             weights = ViT_B_16_Weights.IMAGENET1K_V1
#             print(weights)
#             self.model = vit_b_16(weights=weights)
#         else:
#             self.model = vit_b_16(weights=None)

#         # Modify classifier head to match classes
#         self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)

#         # activations dictionary
#         self.activations = {}

#         # Register hooks
#         self._register_hooks()

#     def _register_hooks(self):
#         # Register forward hooks - encoder blocks
#         for idx, block in enumerate(self.model.encoder.layers):
#             # register_forward_hook(module, input, output) - pytorch built-in function - capture output
#             block.register_forward_hook(self._get_activation(f'encoder_block_{idx}'))
        
#         # extract activation at first

#     def _get_activation(self, name):
#         #  returns a hook that save the activation
#         def hook(module, input, output):
#             self.activations[name] = output.detach() # capture activations
#         return hook

#     def forward(self, x): # activations from hooked layers captured
#         # pass input to model
#         output = self.model(x)
#         return output

#     # retrieves the activations during the pass
#     def forward_features(self, x):
#         self.activations = {} # reset
#         self.forward(x)
#         # collect activations - list of features for each encoder block -> dictionary?
#         features = []
#         for idx in range(len(self.mode.encoder.layers)):
#             activation_extracted = self.activations.get(f'encoder_block_{idx}')
#             if activation_extracted is not None:
#                 featueres.append(activation_extracted)
#         return features

#     def print_weights(self):
#         for name, param in self.model.named_parameters():
#             print(f"Layer: {name}, Weights: {param}")

# if __name__ == "__main__":
#     model = VTransformer(num_classes=10, pretrained=True, input_size=224)
    
#     model.print_weights()





# model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

# # # Print all layers
# for layer in model.modules():
#     print(layer)

# for layer in model.encoder.layers():
#     print(layer)



# def hook_test(module, input, output):
#     print(f"Hook attached to {module}")
#     print(f"Input: {input}")
#     print(f"Output: {output}")

# model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

# first_encoder_layer = model.encoder.layers[0]
# hook = first_encoder_layer.register_forward_hook(hook_test)

# input_tensor = torch.randn(1, 3, 224, 224)
# output = model(input_tensor)

# hook.remove()








# # Using forward_features
# model = VTransformer(num_classes=10, pretrained=True)

# # Dropout removed?
# model.eval()

# example_input = torch.randn(2, 3, 224, 224)

# # Extract features from the encoder blocks using forward_features
# features = model.forward_features(example_input)

# # Print the shape of features from each encoder block
# for i, feature in enumerate(features):
#     print(f"Encoder Block {i} feature shape: {feature.shape}")
