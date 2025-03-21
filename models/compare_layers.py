import argparse
import os
import torch
import numpy as np
from gtda.diagrams import PairwiseDistance
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Compare two ViT models using persistence diagrams.")
    parser.add_argument('--diagram_a', type=str, required=True, help='Path to dgm.pkl for model A.')
    parser.add_argument('--diagram_b', type=str, required=True, help='Path to dgm.pkl for model B.')
    parser.add_argument('--device', type=str, default='cuda', help='Select "cuda" or "cpu".')
    return parser.parse_args()

def load_diagram(path):
    """Load the persistence diagram from a .pkl file."""
    with open(path, 'rb') as f:
        diagrams = pickle.load(f)[0]
    if not isinstance(diagrams, np.ndarray):
        raise ValueError("Expected dgm.pkl to contain a list of diagrams.")
    return diagrams

import numpy as np

def normalize_diagrams(dgm_a, dgm_b, device):
    """Ensures all persistence diagrams have the same number of points."""
    
    max_points = max(max(len(dgm) for dgm in dgm_a), max(len(dgm) for dgm in dgm_b))
    
    def normalize_one_set(dgms):
        normalized_dgms = []

        for dgm in dgms:
            num_points = len(dgm)

            if num_points == max_points:
                normalized_dgms.append(dgm)
                continue

            all_births = torch.tensor(dgm[:, 0], dtype=torch.float32, device=device)
            all_deaths = torch.tensor(dgm[:, 1], dtype=torch.float32, device=device)

            birth_mean, birth_std = torch.mean(all_births), torch.std(all_births)
            death_mean, death_std = torch.mean(all_deaths), torch.std(all_deaths)

            # Consider zero standard deviation?

            # Generate additional points using normal distribution
            num_extra = max_points - num_points
            extra_births = torch.normal(birth_mean, birth_std, size=(num_extra,), device=device)
            extra_deaths = torch.normal(death_mean, death_std, size=(num_extra,), device=device)

            # Ensure birth ≤ death for valid persistence points
            extra_births, extra_deaths = torch.minimum(extra_births, extra_deaths), torch.maximum(extra_births, extra_deaths)

            # Append new points with homology dimension = 0
            extra_points = torch.stack((extra_births, extra_deaths, torch.zeros(num_extra, device=device)), dim=1)
            expanded_dgm = torch.cat((torch.tensor(dgm, dtype=torch.float32, device=device), extra_points), dim=0)
            normalized_dgms.append(expanded_dgm.cpu().numpy())

        return normalized_dgms

    if max(len(dgm) for dgm in dgm_a) < max_points:
        dgm_a = normalize_one_set(dgm_a, max_points)
    if max(len(dgm) for dgm in dgm_b) < max_points:
        dgm_b = normalize_one_set(dgm_b, max_points)

    return dgm_a, dgm_b

def greedy_align_diagrams(dgm_a, dgm_b, dist_metric, device):
    """Greedy alignment of diagrams to find minimized distance."""
    alignment = []
    j_start = 0

    for i, diagram_a in enumerate(dgm_a):
        best_j, best_cost = None, float('inf')
        for j in range(j_start, len(dgm_b)):
            # convert diagram data to float tensors
            diagram_a_tensor = torch.tensor(diagram_a, dtype=torch.float32, device=device).detach()
            diagram_b_tensor = torch.tensor(dgm_b[j], dtype=torch.float32, device=device).detach()
            # diagram_a_tensor = diagram_a.float().to(device).clone().detach()
            # diagram_b_tensor = dgm_b[j].float().to(device).clone().detach()

            # Ensure shape is (N, 3)
            if diagram_a_tensor.shape[-1] != 3 or diagram_b_tensor.shape[-1] != 3:
                raise ValueError(f"Diagrams must have exactly 3 components (birth, death, homology), but got {diagram_a_tensor.shape[-1]} and {diagram_b_tensor.shape[-1]}.")

            dist = dist_metric.fit_transform([diagram_a_tensor.cpu().numpy(), diagram_b_tensor.cpu().numpy()])[0, 1]
            
            if dist < best_cost:
                best_cost = dist
                best_j = j
                
        alignment.append((i, best_j, best_cost))
        j_start = best_j

    return alignment

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load diagrams
    print("\nLoading persistence diagrams...")
    dgm_a = load_diagram(args.diagram_a)
    dgm_b = load_diagram(args.diagram_b)

    # Normalize diagrams - equal sample sizes
    print("\nNormalizing persistence diagrams...")
    dgm_a, dgm_b = normalize_diagrams(dgm_a, dgm_b, device)

    # Use wasserstein metric for comparison
    dist_metric = PairwiseDistance(metric='wasserstein') # bottleneck

    # Greedy alignment of diagrams
    print("\nPerforming greedy alignment of diagrams...")
    alignment = greedy_align_diagrams(dgm_a, dgm_b, dist_metric, device)

    print("\nGreedy Layer Alignment Results:")
    for i, j, cost in alignment:
        print(f"Model A Layer {i} -> Model B Layer {j}, Distance = {cost:.4f}")

    # Save alignment results and paths
    alignment_filename = f"alignment_{os.path.basename(args.diagram_a)}_vs_{os.path.basename(args.diagram_b)}.pkl"
    output_path = os.path.join(os.path.dirname(args.diagram_a), alignment_filename)

    output_data = {
        "alignment": alignment,
        "diagram_a_path": args.diagram_a,
        "diagram_b_path": args.diagram_b
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"\nAlignment results saved to: {output_path}")

if __name__ == "__main__":
    main()