import argparse
import os
import torch
import numpy as np
from PIL import Image
from gtda.diagrams import PairwiseDistance
import pickle

from transformer import (
    VTransformerB16,
    VTransformerB32,
    VTransformerL16,
    VTransformerL32
)

def parse_args():
    parser = argparse.ArgumentParser(description="Compare two ViT models and align their layers.")
    parser.add_argument('--netA', type=str, required=True,
                        help='One of: "b16", "b32", "l16", "l32"')
    parser.add_argument('--netB', type=str, required=True,
                        help='One of: "b16", "b32", "l16", "l32"')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Full path to a single .JPEG image')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Select "cuda" or "cpu"')
    return parser.parse_args()

def build_model(key):
    """
    Build and return the appropriate VTransformer model based on the string.
    """
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
        raise ValueError(f"Unknown model type: {key}. Must be one of b16, b32, l16, l32.")

def greedy_align_layers(b_layers, l_layers, similarity_fn):
    """
    Returns: List of tuples (i, j, cost)
    B-16 layer i is matched with L-16 layer j with a similarity cost.
    """
    alignment = []
    j_start = 0

    for i in range(len(b_layers)):
        best_j, best_cost = None, float('inf')
        # Compare to L-16 layers, starting from the last matched layer.
        for j in range(j_start, len(l_layers)):
            cost = similarity_fn(b_layers[i], l_layers[j])
            if cost < best_cost: # minimum
                best_cost = cost
                best_j = j
        alignment.append((i, best_j, best_cost))  # Store the best match for B-16 layer i.
        j_start = best_j  # Update the start index for L-16 layers
    return alignment

def layer_similarity(tensor1, tensor2):
    """
    Computes similarity between two layers by flattening their activations and calculating L2 norm.
    Returns: L2 distance between the two flattened activations.
    """
    arr1 = tensor1.flatten().cpu().numpy()
    arr2 = tensor2.flatten().cpu().numpy()
    d1 = np.column_stack((arr1, np.zeros_like(arr1), np.zeros_like(arr1)))
    d2 = np.column_stack((arr2, np.zeros_like(arr2), np.zeros_like(arr2)))
    data = np.array([d1, d2], dtype=object) # Stack two persistence diagrams into single array
    dist_metric = PairwiseDistance(metric='euclidean')
    dist_matrix = dist_metric.fit_transform(data)
    return dist_matrix[0, 1]


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    modelA = build_model(args.netA).to(device).eval()
    modelB = build_model(args.netB).to(device).eval()

    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"Image not found: {args.img_path}")

    print(f"Loading image from: {args.img_path}")
    pil_img = Image.open(args.img_path).convert('RGB') # is this needed for imagenet data?

    # Preprocessing transformation of model to the input image, add a batch dimension
    input_tensorA = modelA._get_transform()(pil_img).unsqueeze(0).to(device)
    input_tensorB = modelB._get_transform()(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Extract feature activations
        a_acts = modelA.forward_features(input_tensorA)
        b_acts = modelB.forward_features(input_tensorB)

    alignment = greedy_align_layers(a_acts, b_acts, layer_similarity)

    print(f"\nGreedy alignment of {args.netA.upper()} layers to {args.netB.upper()} layers on: {os.path.basename(args.img_path)}")
    for (i, j, cost) in alignment:
        print(f"  {args.netA.upper()} layer {i} -> {args.netB.upper()} layer {j}, cost = {cost:.4f}")

    image_base = os.path.splitext(os.path.basename(args.img_path))[0]
    alignment_name = f"{args.netA}_{args.netB}_{image_base}.pkl"
    with open(alignment_name, 'wb') as f:
        pickle.dump(alignment, f)
    print(f"\nAlignment saved to: {alignment_name}")

if __name__ == "__main__":
    main()