import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class VTransformer(nn.Module):
    #  simple ViT model that uses pre-trained weights from Torchvision
    def __init__(self, num_classes=10, pretrained=True, img_size=224):
        super(VTransformer, self).__init__()
        # Load pre-trained model
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)

        # Change last layer to match the number of classes
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

    # forward_features,get_features modify
    def forward_features(self, x):
        # extract features from model before the classification head
        with torch.no_grad():
            features = self.vit.forward_features(x)
        return features

def test_vtransformer():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VTransformer(num_classes=10, pretrained=True, img_size=224)

    # model.to(device)
    model.eval()
    example_input = torch.randn(2, 3, 224, 224)
    
    output = model(example_input)
    print(f'output shape: {output.shape}')

    features = model.get_features(example_input)
    print(f'features shape: {features.shape}')

if __name__ == "__main__":
    test_vtransformer()