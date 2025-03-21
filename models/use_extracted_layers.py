import argparse
import pickle
import torch
import numpy as np
from gtda.diagrams import PairwiseDistance

def parse_args():
    parser = argparse.ArgumentParser(description="Compare extracted layers for inference.")
    parser.add_argument('--alignment', type=str, required=True, help='Path to alignment.pkl from compare_layers.py.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on.')
    return parser.parse_args()

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def wasserstein_total_distance(dgm_a, dgm_b, device):
    """Compute Wasserstein distance between full model persistence structures."""
    dist_metric = PairwiseDistance(metric='wasserstein')
    
    dgm_a_tensor = torch.tensor(np.concatenate(dgm_a), dtype=torch.float32).to(device)
    dgm_b_tensor = torch.tensor(np.concatenate(dgm_b), dtype=torch.float32).to(device)

    total_dist = dist_metric.fit_transform([dgm_a_tensor.cpu().numpy(), dgm_b_tensor.cpu().numpy()])[0, 1] # use it as tensor?
    
    return total_dist

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    print("\nLoading alignment results...")
    alignment_data = load_pickle(args.alignment) 

    # Extract paths for diagrams from alignment.pkl
    diagram_a_path = alignment_data["diagram_a_path"]
    diagram_b_path = alignment_data["diagram_b_path"]

    print("\nLoading persistence diagrams...")
    dgm_a = load_pickle(diagram_a_path)
    dgm_b = load_pickle(diagram_b_path)
    
    # Extract most similar layers from 
    alignment = alignment_data["alignment"]
    
    if len(dgm_a) == len(alignment):
        selected_a = dgm_a  # Use Model A as-is
        selected_b = [dgm_b[j] for _, j, _ in alignment]  # Extract aligned layers from Model B
    else:
        selected_a = [dgm_a[i] for i, _, _ in alignment]
        selected_b = dgm_b
    

    print(f"\nUsing {len(selected_a)} matched layers from Model A.")
    print(f"Using {len(selected_b)} matched layers from Model B.\n")

    # Compute Wasserstein distance as the final model similarity metric
    final_wasserstein_dist = wasserstein_total_distance(selected_a, selected_b, device)

    print(f"\nFinal Wasserstein Distance: {final_wasserstein_dist:.4f}")

if __name__ == "__main__":
    main()