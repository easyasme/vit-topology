import torch
import torch.nn as nn
from models.transformer import VTransformer
import matplotlib.pyplot as plt
from PIL import Image

def test_vtransformer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for computation.")

    # Initialize VTransformer model
    num_classes = 10
    model = VTransformer(num_classes=num_classes, pretrained=True, input_size=224)
    model.to(device)
    model.eval()  # set model to evaluation mode - disable dropout
    print(model)

    batch_size = 1
    input_size = 224
    # dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
    dummy_input Image.open('data/imagenet_sample.jpg').resize((input_size, input_size))

    # Run forward pass through the model
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")  # [batch_size, num_classes]

    # Extract features - forward_features
    features = model.forward_features(dummy_input)
    print(f"Number of features extracted: {len(features)}")  # Check if it matches the number of encoder blocks

    # Shape of activations from each encoder block
    for idx, activation in enumerate(features):
        print(f"Encoder Block {idx} activation shape: {activation.shape}")
        plt.imshow(activation.permute(1, 2, 0).numpy(force=True))
        plt.savefig(f"encoder_block_{idx}_activation.png")


    # Shape of activations from each encoder layer norm - need to remove
    if 'encoder_ln' in model.activations:
        encoder_ln_activation = model.activations['encoder_ln']
        print(f"Encoder LayerNorm activation shape: {encoder_ln_activation.shape}")

    # Value of activations from each encoder block
    for idx, activation in enumerate(features):
        print(f"Encoder Block {idx} activation summary:")
        print(f"Mean: {activation.mean().item()}, Std: {activation.std().item()}, Min: {activation.min().item()}, Max: {activation.max().item()}")


if __name__ == "__main__":
    test_vtransformer()

