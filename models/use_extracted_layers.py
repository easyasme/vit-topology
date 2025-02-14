import argparse
import os
import pickle
import torch
import numpy as np
from PIL import Image

from transformer import (
    VTransformerB16,
    VTransformerB32,
    VTransformerL16,
    VTransformerL32
)

def parse_args():
    parser = argparse.ArgumentParser()

def build_model(key):
    """Return the appropriate VTransformer model based on the string."""
    key = key.lower()
    if key == 'b16':
        return VTransformerB16(num_classes=10, pretrained=True)
    elif key == 'b32':
        return VTransformerB32(num_classes=10, pretrained=True)
    elif key == 'l16':
        return VTransformerL16(num_classes=10, pretrained=True)
    elif key == 'l32':
        return VTransformerL32(num_classes=10, pretrained=True)
    else:
        raise ValueError(f"Unknown model type: {key}")

class PrunedModel(torch.nn.Module):
    def __init__(self, model, matched_layers):
        super(PrunedModel, self).__init__()

        self.model = model
        self.matched_layers = matched_layers

    def forward(self, x):
        pass


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load alignment
    if not os.path.exists(args.alignment_pkl):
        raise FileNotFoundError(f"Alignment PKL not found: {args.alignment_pkl}")
    with open(args.alignment_pkl, 'rb') as f:
        alignment = pickle.load(f)

    netA = build_model(args.netA).to(device).eval()
    netB = build_model(args.netB).to(device).eval()

    matched_i = [t[0] for t in alignment]
    matched_j = [t[1] for t in alignment]

    print(f"Original netA has {num_layers_a} total layers.")
    print(f"Original netB has {num_layers_b} total layers.")
    print(f"Alignment length: {len(alignment)}")


    prunedA = netA
    prunedB = netB

    if num_layers_a < num_layers_b:
        prunedB = PrunedModel(netB, matched_j)
    elif num_layers_b < num_layers_a:
        prunedA = PrunedModel(netA, matched_i)
    else:
        print("Both models have the same number of layers. No pruning needed.")

    if args.img_path:
        if not os.path.exists(args.img_path):
            raise FileNotFoundError(f"Image not found: {args.img_path}")
        pil_img = Image.open(args.img_path).convert('RGB')
        transform_func = netA._get_transform()
        input_tensor = transform_func(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outA = prunedA.forward_features(input_tensor)
            outB = prunedB.forward_features(input_tensor)

        print("\nAfter pruning, test inference with single image:")
        print(f" prunedA layers = {len(outA)}   prunedB layers = {len(outB)}")

if __name__ == "__main__":
    main()