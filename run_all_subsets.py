import os
import itertools
import subprocess
import argparse

def find_pkl_files(root_path):
    """ Find all .pkl files in subdirectories. """
    pkl_files = []
    for root, _, files in os.walk(root_path):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    return pkl_files

def generate_subsets(pkl_files):
    """ Generate all pairs of .pkl files for comparison. """
    return list(itertools.combinations(pkl_files, 2))

def run_comparisons(pkl_pairs, compare_script, device):
    """ Run for each pair of .pkl files. """
    for diagram_a, diagram_b in pkl_pairs:
        print(f"\nRunning comparison between:\n  {diagram_a}\n  {diagram_b}")

        cmd = [
            "python", compare_script,
            "--diagram_a", diagram_a,
            "--diagram_b", diagram_b,
            "--device", device
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running comparison: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run all subset comparisons for ViT model persistence diagrams.")
    parser.add_argument('--root_path', type=str, required=True, help="Root directory containing subfolders with .pkl files.")
    parser.add_argument('--compare_script', type=str, default="compare_layers.py", help="Path to compare_layers.py script.")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'.")

    args = parser.parse_args()

    pkl_files = find_pkl_files(args.root_path)

    pkl_pairs = generate_subsets(pkl_files)

    run_comparisons(pkl_pairs, args.compare_script, args.device)

if __name__ == "__main__":
    main()